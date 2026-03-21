/**
 * API client for loan default prediction service.
 * Handles communication with FastAPI backend.
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface LoanApplicationRequest {
  loan_id: string;
  age: number;
  income: number;
  loan_amount: number;
  credit_score: number;
  months_employed: number;
  num_credit_lines: number;
  interest_rate: number;
  loan_term: number;
  dti_ratio: number;
  education: 'High School' | "Bachelor's" | "Master's" | 'PhD';
  employment_type: 'Full-time' | 'Part-time' | 'Self-employed' | 'Unemployed';
  marital_status: 'Single' | 'Married' | 'Divorced';
  has_mortgage: 'Yes' | 'No';
  has_dependents: 'Yes' | 'No';
  loan_purpose: 'Home' | 'Auto' | 'Education' | 'Business' | 'Other';
  has_cosigner: 'Yes' | 'No';
}

export interface PredictionResponse {
  loan_id: string;
  default_probability: number;
  risk_category: 'Low' | 'Medium' | 'High';
  prediction_timestamp: number;
  model_version: string;
}

export interface ExplanationResponse {
  loan_id: string;
  feature_contributions: Record<string, number>;
  base_value: number;
  explanation_timestamp: number;
  model_version: string;
}

class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public response?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Generic API request handler with error handling and retries.
 */
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {},
  retries: number = 3
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const defaultHeaders = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  };

  const config: RequestInit = {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  };

  // Add authentication if available
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers = {
      ...config.headers,
      'Authorization': `Bearer ${token}`,
    };
  }

  let lastError: Error;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new ApiError(
          errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
          response.status,
          errorData
        );
      }

      return await response.json();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      
      // Don't retry on client errors (4xx)
      if (lastError instanceof ApiError && lastError.status && lastError.status < 500) {
        throw lastError;
      }
      
      // Wait before retry (exponential backoff)
      if (attempt < retries) {
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
      }
    }
  }

  throw lastError!;
}

/**
 * Predict loan default probability.
 */
export async function predictLoan(
  application: LoanApplicationRequest
): Promise<PredictionResponse> {
  try {
    const response = await apiRequest<PredictionResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(application),
    });
    
    return response;
  } catch (error) {
    console.error('Prediction failed:', error);
    throw error;
  }
}

/**
 * Get model explanation for a loan application.
 */
export async function explainPrediction(
  application: LoanApplicationRequest
): Promise<ExplanationResponse> {
  try {
    const response = await apiRequest<ExplanationResponse>('/explain', {
      method: 'POST',
      body: JSON.stringify(application),
    });
    
    return response;
  } catch (error) {
    console.error('Explanation failed:', error);
    throw error;
  }
}

/**
 * Batch prediction for multiple applications.
 */
export async function predictBatch(
  applications: LoanApplicationRequest[]
): Promise<PredictionResponse[]> {
  try {
    const response = await apiRequest<PredictionResponse[]>('/predict/batch', {
      method: 'POST',
      body: JSON.stringify({ applications }),
    });
    
    return response;
  } catch (error) {
    console.error('Batch prediction failed:', error);
    throw error;
  }
}

/**
 * Check API health status.
 */
export async function checkHealth(): Promise<{
  status: string;
  timestamp: number;
  model_loaded: boolean;
  preprocessor_loaded: boolean;
  redis_connected: boolean;
}> {
  try {
    const response = await apiRequest('/health');
    return response;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
}

/**
 * Get API metrics (if monitoring is enabled).
 */
export async function getMetrics(): Promise<string> {
  try {
    const response = await fetch(`${API_BASE_URL}/metrics`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return await response.text();
  } catch (error) {
    console.error('Metrics fetch failed:', error);
    throw error;
  }
}

/**
 * Validate loan application data before submission.
 */
export function validateApplication(
  application: Partial<LoanApplicationRequest>
): { isValid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Required fields
  const requiredFields: (keyof LoanApplicationRequest)[] = [
    'loan_id', 'age', 'income', 'loan_amount', 'credit_score',
    'months_employed', 'num_credit_lines', 'interest_rate',
    'loan_term', 'dti_ratio', 'education', 'employment_type',
    'marital_status', 'has_mortgage', 'has_dependents',
    'loan_purpose', 'has_cosigner'
  ];

  for (const field of requiredFields) {
    if (!application[field]) {
      errors.push(`${field.replace(/_/g, ' ')} is required`);
    }
  }

  // Business logic validations
  if (application.age && (application.age < 18 || application.age > 100)) {
    errors.push('Age must be between 18 and 100');
  }

  if (application.income && application.income < 15000) {
    errors.push('Income must be at least $15,000');
  }

  if (application.credit_score && (application.credit_score < 300 || application.credit_score > 850)) {
    errors.push('Credit score must be between 300 and 850');
  }

  if (application.dti_ratio && application.dti_ratio > 0.8) {
    errors.push('Debt-to-income ratio cannot exceed 80%');
  }

  if (application.loan_amount && application.income && 
      application.loan_amount > application.income * 10) {
    errors.push('Loan amount cannot exceed 10x annual income');
  }

  return {
    isValid: errors.length === 0,
    errors
  };
}

/**
 * Format currency values for display.
 */
export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount);
}

/**
 * Format percentage values for display.
 */
export function formatPercentage(value: number, decimals: number = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Generate unique loan ID for new applications.
 */
export function generateLoanId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substr(2, 5);
  return `LN${timestamp}${random}`.toUpperCase();
}

/**
 * Calculate loan monthly payment (simple approximation).
 */
export function calculateMonthlyPayment(
  principal: number,
  annualRate: number,
  months: number
): number {
  const monthlyRate = annualRate / 100 / 12;
  if (monthlyRate === 0) {
    return principal / months;
  }
  
  const payment = principal * 
    (monthlyRate * Math.pow(1 + monthlyRate, months)) /
    (Math.pow(1 + monthlyRate, months) - 1);
  
  return Math.round(payment);
}

/**
 * Determine if a loan application should be flagged for manual review.
 */
export function shouldFlagForReview(
  prediction: PredictionResponse
): { shouldFlag: boolean; reason?: string } {
  if (prediction.default_probability > 0.7) {
    return { shouldFlag: true, reason: 'High default probability' };
  }
  
  if (prediction.risk_category === 'High') {
    return { shouldFlag: true, reason: 'High risk category' };
  }
  
  return { shouldFlag: false };
}

/**
 * Cache API responses in localStorage for better UX.
 */
class ApiCache {
  private static instance: ApiCache;
  private cache: Map<string, { data: any; timestamp: number; ttl: number }>;

  private constructor() {
    this.cache = new Map();
  }

  static getInstance(): ApiCache {
    if (!ApiCache.instance) {
      ApiCache.instance = new ApiCache();
    }
    return ApiCache.instance;
  }

  set(key: string, data: any, ttl: number = 300000): void { // 5 minutes default TTL
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    });
  }

  get(key: string): any | null {
    const item = this.cache.get(key);
    if (!item) return null;

    if (Date.now() - item.timestamp > item.ttl) {
      this.cache.delete(key);
      return null;
    }

    return item.data;
  }

  clear(): void {
    this.cache.clear();
  }
}

export const apiCache = ApiCache.getInstance();

/**
 * Wrapper functions with caching.
 */
export async function predictLoanWithCache(
  application: LoanApplicationRequest
): Promise<PredictionResponse> {
  const cacheKey = `predict_${JSON.stringify(application)}`;
  
  // Try cache first
  const cached = apiCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  // Make API call
  const result = await predictLoan(application);
  
  // Cache result
  apiCache.set(cacheKey, result, 60000); // 1 minute cache
  
  return result;
}
