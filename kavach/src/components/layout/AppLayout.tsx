import { useState, useEffect } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { Moon, Sun } from 'lucide-react';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const navItems = [
  { path: '/', label: 'Overview' },
  { path: '/enrolment', label: 'Enrolment' },
  { path: '/authenticate', label: 'Authenticate' },
  { path: '/speakers', label: 'Speakers' },
  { path: '/graph-explorer', label: 'Graph Explorer' },
  { path: '/attack-lab', label: 'Attack Lab' },
  { path: '/evaluation', label: 'Evaluation' },
  { path: '/corpus', label: 'Corpus' },
];

function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    if (typeof window !== 'undefined') {
      return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
    }
    return 'light';
  });

  const toggleTheme = () => {
    const newTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    if (newTheme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('theme', newTheme);
  };

  return { theme, toggleTheme };
}

export function AppLayout() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: apiClient.health,
    retry: false
  });

  const { theme, toggleTheme } = useTheme();

  return (
    <div className="flex h-screen bg-app-bg text-app-text overflow-hidden">
      {/* Sidebar */}
      <aside className="w-[200px] flex-shrink-0 border-r border-app-border flex flex-col">
        <div className="p-4 border-b border-app-border">
          <h1 className="text-[14px] font-semibold tracking-tight">KAVACH</h1>
          <p className="text-[10px] mono opacity-50 mt-1">v0.1 · research build</p>
        </div>

        <nav className="flex-1 py-4 flex flex-col gap-0.5">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                cn(
                  "px-4 py-2 text-[13px] border-l-[2px] transition-colors duration-120 cursor-pointer",
                  isActive ? "border-app-accent text-app-accent font-medium" : "border-transparent text-app-text opacity-60 hover:opacity-100"
                )
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-app-border bg-app-surface-muted">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className={cn("w-1.5 h-1.5 rounded-full", health ? "bg-app-accept" : "bg-app-reject")} />
              <span className="text-[10px] font-mono uppercase tracking-wider">{health ? 'Connected' : 'Offline'}</span>
            </div>
            <button 
              onClick={toggleTheme}
              className="text-app-text-muted hover:text-app-text transition-colors"
              title="Toggle Theme"
            >
              {theme === 'dark' ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
            </button>
          </div>
          {health && (
            <div className="text-[10px] font-mono opacity-60 leading-relaxed">
              MODEL: {health.models[0].toUpperCase()}<br/>
              DEVICE: {health.device.toUpperCase()}
            </div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 bg-app-bg overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
