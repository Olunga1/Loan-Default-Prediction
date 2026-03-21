"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { LoanForm } from '@/components/forms/LoanForm';
import { RiskChart } from '@/components/charts/RiskChart';
import { FeatureImportanceChart } from '@/components/charts/FeatureImportanceChart';
import { PredictionResponse, ExplanationResponse } from '@/types';
import { predictLoan, explainPrediction } from '@/lib/api';
import { AlertTriangle, CheckCircle, Clock, TrendingUp } from 'lucide-react';

export default function Home() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recentPredictions, setRecentPredictions] = useState<PredictionResponse[]>([]);

  const handlePrediction = async (formData: any) => {
    setLoading(true);
    setError(null);
    
    try {
      // Make prediction
      const predictionResult = await predictLoan(formData);
      setPrediction(predictionResult);
      
      // Get explanation
      const explanationResult = await explainPrediction(formData);
      setExplanation(explanationResult);
      
      // Add to recent predictions
      setRecentPredictions(prev => [predictionResult, ...prev.slice(0, 4)]);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (riskCategory: string) => {
    switch (riskCategory) {
      case 'Low': return 'bg-green-100 text-green-800';
      case 'Medium': return 'bg-yellow-100 text-yellow-800';
      case 'High': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getRiskIcon = (riskCategory: string) => {
    switch (riskCategory) {
      case 'Low': return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'Medium': return <Clock className="w-5 h-5 text-yellow-600" />;
      case 'High': return <AlertTriangle className="w-5 h-5 text-red-600" />;
      default: return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Loan Default Risk Assessment
          </h1>
          <p className="text-lg text-gray-600">
            AI-powered credit risk analysis with real-time predictions and explainability
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Loan Application Form */}
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle>Loan Application</CardTitle>
                <CardDescription>
                  Enter applicant information to assess default risk
                </CardDescription>
              </CardHeader>
              <CardContent>
                <LoanForm onSubmit={handlePrediction} loading={loading} />
              </CardContent>
            </Card>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-2 space-y-6">
            {error && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {prediction && (
              <>
                {/* Prediction Result */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      Risk Assessment Result
                      <Badge className={getRiskColor(prediction.risk_category)}>
                        {getRiskIcon(prediction.risk_category)}
                        {prediction.risk_category} Risk
                      </Badge>
                    </CardTitle>
                    <CardDescription>
                      Loan ID: {prediction.loan_id}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="text-center">
                        <div className="text-3xl font-bold text-blue-600">
                          {(prediction.default_probability * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-gray-600">Default Probability</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-semibold text-gray-900">
                          {prediction.risk_category}
                        </div>
                        <div className="text-sm text-gray-600">Risk Category</div>
                      </div>
                      <div className="text-center">
                        <div className="text-sm text-gray-500">
                          Model v{prediction.model_version}
                        </div>
                        <div className="text-sm text-gray-600">
                          {new Date(prediction.prediction_timestamp * 1000).toLocaleString()}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Risk Visualization */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="w-5 h-5" />
                      Risk Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <RiskChart 
                      probability={prediction.default_probability}
                      riskCategory={prediction.risk_category}
                    />
                  </CardContent>
                </Card>

                {/* Feature Importance */}
                {explanation && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Feature Contributions</CardTitle>
                      <CardDescription>
                        Key factors influencing this prediction
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <FeatureImportanceChart 
                        contributions={explanation.feature_contributions}
                      />
                    </CardContent>
                  </Card>
                )}
              </>
            )}

            {/* Recent Predictions */}
            {recentPredictions.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Recent Assessments</CardTitle>
                  <CardDescription>
                    Latest loan risk evaluations
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {recentPredictions.map((pred, index) => (
                      <div key={`${pred.loan_id}-${index}`} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center gap-3">
                          <Badge className={getRiskColor(pred.risk_category)}>
                            {pred.risk_category}
                          </Badge>
                          <span className="text-sm font-medium">{pred.loan_id}</span>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-semibold">
                            {(pred.default_probability * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-gray-500">
                            {new Date(pred.prediction_timestamp * 1000).toLocaleTimeString()}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>

        {/* Statistics Overview */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-8 h-8 text-green-600" />
                <div>
                  <div className="text-2xl font-bold text-green-600">87%</div>
                  <div className="text-sm text-gray-600">Low Risk Applications</div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Clock className="w-8 h-8 text-yellow-600" />
                <div>
                  <div className="text-2xl font-bold text-yellow-600">11%</div>
                  <div className="text-sm text-gray-600">Medium Risk Applications</div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-8 h-8 text-red-600" />
                <div>
                  <div className="text-2xl font-bold text-red-600">2%</div>
                  <div className="text-sm text-gray-600">High Risk Applications</div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-8 h-8 text-blue-600" />
                <div>
                  <div className="text-2xl font-bold text-blue-600">94ms</div>
                  <div className="text-sm text-gray-600">Avg Response Time</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
