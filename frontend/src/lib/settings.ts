const STORAGE_KEY = 'face-recognition-settings';

export interface AppSettings {
  selectedModel: string;
  threshold: number;
}

const defaultSettings: AppSettings = {
  selectedModel: 'CustomFaceNet',
  threshold: 0.7,
};

export function getSettings(): AppSettings {
  if (typeof window === 'undefined') return defaultSettings;
  
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return { ...defaultSettings, ...JSON.parse(stored) };
    }
  } catch (e) {
    console.error('Failed to load settings:', e);
  }
  return defaultSettings;
}

export function saveSettings(settings: Partial<AppSettings>): AppSettings {
  if (typeof window === 'undefined') return { ...defaultSettings, ...settings };
  
  try {
    const current = getSettings();
    const updated = { ...current, ...settings };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    return updated;
  } catch (e) {
    console.error('Failed to save settings:', e);
    return { ...defaultSettings, ...settings };
  }
}

export function useSettings() {
  return { getSettings, saveSettings };
}
