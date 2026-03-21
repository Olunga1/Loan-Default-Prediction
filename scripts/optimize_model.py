#!/usr/bin/env python3
"""
Model optimization script for production deployment.
Includes ONNX export, quantization, and performance benchmarking.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import joblib
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
import psutil
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessing import FeaturePreprocessor
from src.utils.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Optimize trained model for production deployment."""
    
    def __init__(self, model_path: str, preprocessor_path: str, output_dir: str):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and preprocessor
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        
        # Initialize ONNX runtime session
        self.onnx_session = None
        
    def export_to_onnx(self, sample_input: np.ndarray) -> str:
        """Export XGBoost model to ONNX format."""
        logger.info("Exporting model to ONNX format...")
        
        onnx_path = self.output_dir / "model.onnx"
        
        # Create initial types for ONNX
        initial_type = [('float_input', onnx.TensorType(sample_input.shape))]
        
        # Export XGBoost model to ONNX
        onnx_model = xgb.convert_to_onnx(self.model, initial_types=initial_type)
        
        # Save ONNX model
        onnx.save_model(onnx_model, str(onnx_path))
        
        # Verify ONNX model
        onnx.checker.check_model(str(onnx_path))
        
        logger.info(f"ONNX model saved to {onnx_path}")
        return str(onnx_path)
    
    def quantize_model(self, onnx_path: str) -> str:
        """Apply dynamic quantization to reduce model size and improve inference speed."""
        logger.info("Applying dynamic quantization...")
        
        quantized_path = self.output_dir / "model_quantized.onnx"
        
        # Apply dynamic quantization
        quantize_dynamic(
            str(onnx_path),
            str(quantized_path),
            weight_type=QuantType.QUInt8
        )
        
        logger.info(f"Quantized model saved to {quantized_path}")
        return str(quantized_path)
    
    def create_inference_session(self, model_path: str, providers: list = None) -> ort.InferenceSession:
        """Create ONNX Runtime inference session with specified providers."""
        if providers is None:
            # Use CPU by default, GPU if available
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(model_path, providers=providers)
        logger.info(f"Created inference session with providers: {providers}")
        return session
    
    def benchmark_model(self, 
                       session: ort.InferenceSession, 
                       test_data: np.ndarray,
                       num_runs: int = 100) -> Dict[str, float]:
        """Benchmark model inference performance."""
        logger.info(f"Benchmarking model with {num_runs} runs...")
        
        # Warm up
        for _ in range(10):
            session.run(None, {'float_input': test_data})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = session.run(None, {'float_input': test_data})
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        p99_time = np.percentile(times, 99)
        min_time = np.min(times)
        max_time = np.max(times)
        
        stats = {
            'avg_time_ms': avg_time * 1000,
            'p95_time_ms': p95_time * 1000,
            'p99_time_ms': p99_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'throughput_qps': 1.0 / avg_time
        }
        
        logger.info(f"Benchmark results: {stats}")
        return stats
    
    def compare_models(self, test_data: pd.DataFrame, test_labels: pd.Series) -> Dict[str, Any]:
        """Compare original and optimized models."""
        logger.info("Comparing model performance...")
        
        # Preprocess test data
        X_test_processed = self.preprocessor.transform(test_data)
        
        # Original model predictions
        start_time = time.perf_counter()
        original_proba = self.model.predict_proba(X_test_processed)[:, 1]
        original_time = time.perf_counter() - start_time
        
        # ONNX model predictions
        onnx_path = self.export_to_onnx(X_test_processed[:1])
        onnx_session = self.create_inference_session(onnx_path)
        
        start_time = time.perf_counter()
        onnx_proba = []
        for i in range(len(X_test_processed)):
            input_data = X_test_processed[i:i+1].astype(np.float32)
            pred = onnx_session.run(None, {'float_input': input_data})[0]
            onnx_proba.append(pred[0, 1])
        onnx_proba = np.array(onnx_proba)
        onnx_time = time.perf_counter() - start_time
        
        # Quantized model predictions
        quantized_path = self.quantize_model(onnx_path)
        quantized_session = self.create_inference_session(quantized_path)
        
        start_time = time.perf_counter()
        quantized_proba = []
        for i in range(len(X_test_processed)):
            input_data = X_test_processed[i:i+1].astype(np.float32)
            pred = quantized_session.run(None, {'float_input': input_data})[0]
            quantized_proba.append(pred[0, 1])
        quantized_proba = np.array(quantized_proba)
        quantized_time = time.perf_counter() - start_time
        
        # Calculate metrics
        original_auc = roc_auc_score(test_labels, original_proba)
        onnx_auc = roc_auc_score(test_labels, onnx_proba)
        quantized_auc = roc_auc_score(test_labels, quantized_proba)
        
        # Calculate model sizes
        original_size = self.model_path.stat().st_size / (1024 * 1024)  # MB
        onnx_size = Path(onnx_path).stat().st_size / (1024 * 1024)
        quantized_size = Path(quantized_path).stat().st_size / (1024 * 1024)
        
        comparison = {
            'original': {
                'auc': original_auc,
                'inference_time_ms': original_time * 1000,
                'model_size_mb': original_size
            },
            'onnx': {
                'auc': onnx_auc,
                'inference_time_ms': onnx_time * 1000,
                'model_size_mb': onnx_size,
                'speedup': original_time / (onnx_time / len(X_test_processed)),
                'size_reduction': original_size / onnx_size
            },
            'quantized': {
                'auc': quantized_auc,
                'inference_time_ms': quantized_time * 1000,
                'model_size_mb': quantized_size,
                'speedup': original_time / (quantized_time / len(X_test_processed)),
                'size_reduction': original_size / quantized_size,
                'accuracy_loss': original_auc - quantized_auc
            }
        }
        
        logger.info(f"Model comparison: {comparison}")
        return comparison
    
    def optimize_for_production(self, test_data: pd.DataFrame, test_labels: pd.Series):
        """Complete optimization pipeline for production."""
        logger.info("Starting production optimization pipeline...")
        
        # Preprocess test data
        X_test_processed = self.preprocessor.transform(test_data)
        
        # Export to ONNX
        onnx_path = self.export_to_onnx(X_test_processed[:1])
        
        # Quantize model
        quantized_path = self.quantize_model(onnx_path)
        
        # Create inference sessions
        onnx_session = self.create_inference_session(onnx_path)
        quantized_session = self.create_inference_session(quantized_path)
        
        # Benchmark models
        onnx_stats = self.benchmark_model(onnx_session, X_test_processed[:1])
        quantized_stats = self.benchmark_model(quantized_session, X_test_processed[:1])
        
        # Compare models
        comparison = self.compare_models(test_data, test_labels)
        
        # Generate optimization report
        report = {
            'optimization_timestamp': time.time(),
            'model_comparison': comparison,
            'benchmark_stats': {
                'onnx': onnx_stats,
                'quantized': quantized_stats
            },
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version
            },
            'recommendations': self._generate_recommendations(comparison)
        }
        
        # Save report
        report_path = self.output_dir / "optimization_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Optimization report saved to {report_path}")
        return report
    
    def _generate_recommendations(self, comparison: Dict[str, Any]) -> list:
        """Generate optimization recommendations based on comparison results."""
        recommendations = []
        
        # Speed recommendations
        onnx_speedup = comparison['onnx']['speedup']
        if onnx_speedup < 2.0:
            recommendations.append("Consider GPU acceleration for better inference speed")
        
        # Size recommendations
        quantized_size_reduction = comparison['quantized']['size_reduction']
        if quantized_size_reduction > 2.0:
            recommendations.append("Use quantized model for production to reduce memory footprint")
        
        # Accuracy recommendations
        accuracy_loss = comparison['quantized']['accuracy_loss']
        if accuracy_loss > 0.01:
            recommendations.append("Monitor accuracy loss with quantization - consider mixed precision")
        
        # General recommendations
        recommendations.extend([
            "Implement model versioning for A/B testing",
            "Set up monitoring for inference latency and accuracy",
            "Consider batch inference for better throughput",
            "Implement model caching for frequently used predictions"
        ])
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description="Optimize model for production")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--preprocessor-path", required=True, help="Path to preprocessor")
    parser.add_argument("--output-dir", required=True, help="Output directory for optimized models")
    parser.add_argument("--test-data", required=True, help="Path to test data")
    parser.add_argument("--test-labels", required=True, help="Path to test labels")
    
    args = parser.parse_args()
    
    # Load test data
    test_data = pd.read_csv(args.test_data)
    test_labels = pd.read_csv(args.test_labels).squeeze()
    
    # Initialize optimizer
    optimizer = ModelOptimizer(args.model_path, args.preprocessor_path, args.output_dir)
    
    # Run optimization
    report = optimizer.optimize_for_production(test_data, test_labels)
    
    # Print summary
    print("\nOptimization Summary:")
    print(f"ONNX speedup: {report['model_comparison']['onnx']['speedup']:.2f}x")
    print(f"Quantized speedup: {report['model_comparison']['quantized']['speedup']:.2f}x")
    print(f"Size reduction (quantized): {report['model_comparison']['quantized']['size_reduction']:.2f}x")
    print(f"Accuracy loss (quantized): {report['model_comparison']['quantized']['accuracy_loss']:.4f}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")

if __name__ == "__main__":
    main()
