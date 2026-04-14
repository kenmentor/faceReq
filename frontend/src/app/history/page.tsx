"use client";

import { useState, useEffect, useCallback } from "react";
import { apiClient, HistoryEntry } from "@/lib/api";

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [searchName, setSearchName] = useState("");
  const [filterModel, setFilterModel] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [message, setMessage] = useState<string | null>(null);

  const loadHistory = useCallback(async () => {
    setIsLoading(true);
    setMessage(null);
    try {
      const entries = await apiClient.getHistory(
        searchName || undefined,
        filterModel || undefined
      );
      setHistory(entries);
      if (entries.length === 0) {
        setMessage("No verification history");
      }
    } catch {
      setMessage("Failed to load history");
    } finally {
      setIsLoading(false);
    }
  }, [searchName, filterModel]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
  };

  const uniqueModels = Array.from(new Set(history.map((h) => h.model)));

  return (
    <div style={{ background: 'var(--background)', minHeight: 'calc(100vh - 57px)' }}>
      <div className="max-w-lg mx-auto px-4 py-6">
        <h1 style={{ color: 'var(--text-primary)' }} className="text-2xl font-bold mb-6">History</h1>

        <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
          <input
            type="text"
            value={searchName}
            onChange={(e) => setSearchName(e.target.value)}
            placeholder="Search by name"
            style={{
              flex: 1,
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: '8px',
              padding: '10px 12px',
              color: 'var(--text-primary)',
              fontSize: '15px'
            }}
          />
          <select
            value={filterModel}
            onChange={(e) => setFilterModel(e.target.value)}
            style={{
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: '8px',
              padding: '10px 12px',
              color: 'var(--text-primary)',
              fontSize: '15px'
            }}
          >
            <option value="">All</option>
            {uniqueModels.map((model) => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </div>

        {isLoading ? (
          <div style={{ textAlign: 'center', padding: '48px', color: 'var(--text-secondary)' }}>
            Loading...
          </div>
        ) : message ? (
          <div style={{ textAlign: 'center', padding: '48px', color: 'var(--text-secondary)' }}>
            {message}
          </div>
        ) : (
          <div className="space-y-2">
            {history.map((entry) => (
              <div
                key={entry.id}
                style={{
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: '12px',
                  padding: '16px'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{
                    width: '48px',
                    height: '48px',
                    borderRadius: '50%',
                    background: entry.result.is_match ? 'rgba(0, 186, 124, 0.2)' : 'var(--border)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '20px'
                  }}>
                    {entry.result.is_match ? '✓' : '?'}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ color: 'var(--text-primary)', fontWeight: '600' }}>
                        {entry.result.is_match ? entry.result.name : 'Unknown'}
                      </span>
                      <span style={{ color: 'var(--text-tertiary)', fontSize: '14px' }}>
                        {formatDate(entry.timestamp)}
                      </span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '4px' }}>
                      <span style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
                        {(entry.result.confidence * 100).toFixed(0)}% match
                      </span>
                      <span style={{ color: 'var(--text-tertiary)', fontSize: '12px' }}>
                        {entry.model}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
