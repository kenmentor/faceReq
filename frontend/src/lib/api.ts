import axios from 'axios';
import { getSettings } from './settings';

const API_BASE_URL = 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

export interface ModelInfo {
  name: string;
  display_name: string;
  available: boolean;
}

export interface VerificationResult {
  name: string;
  confidence: number;
  is_match: boolean;
  model: string;
}

export interface HistoryEntry {
  id: string;
  timestamp: string;
  result: {
    name: string;
    confidence: number;
    is_match: boolean;
  };
  model: string;
  input_method: string;
  threshold: number;
}

export interface UserInfo {
  id: string;
  name: string;
  enrolled_at: string;
}

export const apiClient = {
  async getModels(): Promise<ModelInfo[]> {
    const response = await api.get<ModelInfo[]>('/models');
    return response.data;
  },

  async enrollUser(name: string, images: File[]): Promise<any> {
    const formData = new FormData();
    formData.append('name', name);
    images.forEach((image) => {
      formData.append('files', image);
    });
    const response = await api.post('/enroll', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async verifyFace(image: File, model?: string, threshold?: number): Promise<VerificationResult> {
    const settings = getSettings();
    const formData = new FormData();
    formData.append('file', image);
    formData.append('model', model || settings.selectedModel);
    formData.append('threshold', (threshold ?? settings.threshold).toString());
    const response = await api.post<VerificationResult>('/verify', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async getHistory(
    name?: string,
    model?: string,
    startDate?: string,
    endDate?: string
  ): Promise<HistoryEntry[]> {
    const params: Record<string, string> = {};
    if (name) params.name = name;
    if (model) params.model = model;
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;
    const response = await api.get<HistoryEntry[]>('/history', { params });
    return response.data;
  },

  async getUsers(): Promise<UserInfo[]> {
    const response = await api.get<UserInfo[]>('/users');
    return response.data;
  },

  async deleteUser(userId: string): Promise<void> {
    await api.delete(`/user/${userId}`);
  },
};

export default apiClient;
