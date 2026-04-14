"use client";

import { useState, useEffect } from "react";
import { apiClient, ModelInfo, UserInfo } from "@/lib/api";
import { getSettings, saveSettings } from "@/lib/settings";

export default function SettingsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [users, setUsers] = useState<UserInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("CustomFaceNet");
  const [threshold, setThreshold] = useState(0.7);
  const [isLoading, setIsLoading] = useState(true);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [deletingUser, setDeletingUser] = useState<string | null>(null);

  useEffect(() => {
    const settings = getSettings();
    setSelectedModel(settings.selectedModel);
    setThreshold(settings.threshold);
    loadData();
  }, []);

  const loadData = async () => {
    setIsLoading(true);
    try {
      const [modelList, userList] = await Promise.all([
        apiClient.getModels(),
        apiClient.getUsers(),
      ]);
      setModels(modelList);
      setUsers(userList);
    } catch {
      setMessage({ type: "error", text: "Failed to load data" });
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelChange = (model: string) => {
    setSelectedModel(model);
    saveSettings({ selectedModel: model });
    setMessage({ type: "success", text: "Settings saved" });
    setTimeout(() => setMessage(null), 2000);
  };

  const handleThresholdChange = (value: number) => {
    setThreshold(value);
    saveSettings({ threshold: value });
    setMessage({ type: "success", text: "Settings saved" });
    setTimeout(() => setMessage(null), 2000);
  };

  const handleDeleteUser = async (userId: string, userName: string) => {
    if (!confirm(`Delete ${userName}?`)) return;
    
    setDeletingUser(userId);
    try {
      await apiClient.deleteUser(userId);
      setUsers(prev => prev.filter(u => u.id !== userId));
      setMessage({ type: "success", text: "User deleted" });
      setTimeout(() => setMessage(null), 2000);
    } catch {
      setMessage({ type: "error", text: "Failed to delete user" });
    } finally {
      setDeletingUser(null);
    }
  };

  return (
    <div style={{ background: 'var(--background)', minHeight: 'calc(100vh - 57px)' }}>
      <div className="max-w-lg mx-auto px-4 py-6">
        <h1 style={{ color: 'var(--text-primary)' }} className="text-2xl font-bold mb-6">Settings</h1>

        {message && (
          <div style={{
            padding: '12px 16px',
            borderRadius: '12px',
            background: message.type === 'success' ? 'rgba(0, 186, 124, 0.1)' : 'rgba(244, 33, 46, 0.1)',
            color: message.type === 'success' ? 'var(--success)' : 'var(--error)',
            fontSize: '14px',
            marginBottom: '16px'
          }}>
            {message.text}
          </div>
        )}

        <div style={{ marginBottom: '24px' }}>
          <h2 style={{ color: 'var(--text-primary)', fontWeight: '600', marginBottom: '12px' }}>Preferences</h2>
          
          <div style={{ background: 'var(--surface)', borderRadius: '12px', padding: '16px', border: '1px solid var(--border)' }}>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ color: 'var(--text-secondary)', fontSize: '14px', display: 'block', marginBottom: '8px' }}>
                Default Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => handleModelChange(e.target.value)}
                style={{
                  width: '100%',
                  background: 'var(--background)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  padding: '12px',
                  color: 'var(--text-primary)',
                  fontSize: '15px'
                }}
              >
                {models.map((model) => (
                  <option key={model.name} value={model.name} disabled={!model.available}>
                    {model.display_name} {!model.available && '(unavailable)'}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '8px' }}>
                <span>Default Threshold</span>
                <span>{(threshold * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={threshold}
                onChange={(e) => handleThresholdChange(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent)' }}
              />
            </div>
          </div>
        </div>

        <div style={{ marginBottom: '24px' }}>
          <h2 style={{ color: 'var(--text-primary)', fontWeight: '600', marginBottom: '12px' }}>
            Models ({models.filter(m => m.available).length} available)
          </h2>
          
          <div className="space-y-2">
            {models.map((model) => (
              <div
                key={model.name}
                style={{
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: '12px',
                  padding: '14px 16px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between'
                }}
              >
                <span style={{ color: 'var(--text-primary)', fontWeight: '500' }}>
                  {model.display_name}
                </span>
                <span style={{
                  fontSize: '12px',
                  padding: '4px 10px',
                  borderRadius: '9999px',
                  background: model.available ? 'rgba(0, 186, 124, 0.15)' : 'var(--border)',
                  color: model.available ? 'var(--success)' : 'var(--text-secondary)'
                }}>
                  {model.available ? 'Active' : 'Unavailable'}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h2 style={{ color: 'var(--text-primary)', fontWeight: '600', marginBottom: '12px' }}>
            Enrolled Users ({users.length})
          </h2>
          
          {isLoading ? (
            <div style={{ color: 'var(--text-secondary)', padding: '24px', textAlign: 'center' }}>
              Loading...
            </div>
          ) : users.length === 0 ? (
            <div style={{ color: 'var(--text-secondary)', padding: '24px', textAlign: 'center', background: 'var(--surface)', borderRadius: '12px', border: '1px solid var(--border)' }}>
              No users enrolled
            </div>
          ) : (
            <div className="space-y-2">
              {users.map((user) => (
                <div
                  key={user.id}
                  style={{
                    background: 'var(--surface)',
                    border: '1px solid var(--border)',
                    borderRadius: '12px',
                    padding: '14px 16px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between'
                  }}
                >
                  <div>
                    <span style={{ color: 'var(--text-primary)', fontWeight: '500' }}>{user.name}</span>
                    <span style={{ color: 'var(--text-tertiary)', fontSize: '13px', marginLeft: '8px' }}>
                      {new Date(user.enrolled_at).toLocaleDateString()}
                    </span>
                  </div>
                  <button
                    onClick={() => handleDeleteUser(user.id, user.name)}
                    disabled={deletingUser === user.id}
                    style={{
                      background: 'transparent',
                      color: deletingUser === user.id ? 'var(--text-tertiary)' : 'var(--error)',
                      border: 'none',
                      padding: '8px 12px',
                      fontSize: '14px',
                      cursor: deletingUser === user.id ? 'not-allowed' : 'pointer'
                    }}
                  >
                    {deletingUser === user.id ? 'Deleting...' : 'Delete'}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        <div style={{ marginTop: '32px', padding: '16px', background: 'var(--surface)', borderRadius: '12px', border: '1px solid var(--border)' }}>
          <h3 style={{ color: 'var(--text-secondary)', fontSize: '13px', marginBottom: '8px' }}>System</h3>
          <p style={{ color: 'var(--text-tertiary)', fontSize: '13px' }}>API: localhost:8000</p>
          <p style={{ color: 'var(--text-tertiary)', fontSize: '13px' }}>Detection: MediaPipe</p>
          <p style={{ color: 'var(--text-tertiary)', fontSize: '13px' }}>Storage: JSON Files</p>
        </div>
      </div>
    </div>
  );
}
