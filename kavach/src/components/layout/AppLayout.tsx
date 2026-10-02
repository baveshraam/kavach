import { useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { cn } from '../ui/kit';
import {
  Moon, Sun, LayoutDashboard, UserPlus, ShieldCheck, Users, Network,
  Swords, LineChart, Database, Mic, type LucideIcon,
} from 'lucide-react';

export { cn };

const navGroups: { label: string; items: { path: string; label: string; icon: LucideIcon }[] }[] = [
  {
    label: 'Workspace',
    items: [{ path: '/', label: 'Overview', icon: LayoutDashboard }],
  },
  {
    label: 'Live system',
    items: [
      { path: '/authenticate', label: 'Authenticate', icon: ShieldCheck },
      { path: '/enrolment', label: 'Enrol a speaker', icon: UserPlus },
    ],
  },
  {
    label: 'Data',
    items: [
      { path: '/speakers', label: 'Speakers', icon: Users },
      { path: '/graph-explorer', label: 'Graph Explorer', icon: Network },
      { path: '/corpus', label: 'Corpus', icon: Database },
      { path: '/studio', label: 'Recording Studio', icon: Mic },
    ],
  },
  {
    label: 'Research',
    items: [
      { path: '/attack-lab', label: 'Attack Lab', icon: Swords },
      { path: '/evaluation', label: 'Evaluation', icon: LineChart },
    ],
  },
];

function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() =>
    typeof document !== 'undefined' && document.documentElement.classList.contains('dark') ? 'dark' : 'light',
  );
  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    document.documentElement.classList.toggle('dark', next === 'dark');
    localStorage.setItem('theme', next);
  };
  return { theme, toggleTheme };
}

/** Short, human names for the model strings `/api/health` reports. */
function modelLabel(m: string) {
  if (m.startsWith('faster-whisper')) return { role: 'ASR', name: m.split('/')[1] ? `Whisper ${m.split('/')[1]}` : 'Whisper' };
  if (m.includes('ecapa')) return { role: 'Voiceprint', name: 'ECAPA-TDNN' };
  if (m.includes('LaBSE')) return { role: 'Matcher', name: 'LaBSE' };
  if (m.startsWith('gemini') || m.startsWith('anthropic') || m.startsWith('openai')) {
    return { role: 'Tagger', name: m.split('/')[1]?.split(' ')[0] ?? m };
  }
  return { role: 'Model', name: m };
}

export function AppLayout() {
  const { data: health, isError } = useQuery({
    queryKey: ['health'],
    queryFn: apiClient.health,
    retry: false,
    refetchInterval: 30000,
  });
  const { theme, toggleTheme } = useTheme();
  const location = useLocation();

  const status = !health || isError ? 'offline' : health.status === 'degraded' ? 'degraded' : 'connected';

  return (
    <div className="flex h-screen bg-app-bg text-app-text overflow-hidden">
      <aside className="w-[240px] shrink-0 border-r border-app-border bg-app-surface flex flex-col">
        {/* Brand */}
        <div className="h-16 px-5 flex items-center gap-3 border-b border-app-border">
          <div className="w-8 h-8 rounded-md bg-app-text text-app-bg flex items-center justify-center font-serif text-[17px] font-semibold leading-none">
            க
          </div>
          <div className="leading-tight">
            <div className="text-[15px] font-semibold tracking-tight">KAVACH</div>
            <div className="text-[11.5px] text-app-text-subtle">Voice authentication</div>
          </div>
        </div>

        <nav className="flex-1 overflow-y-auto px-3 py-4 flex flex-col gap-5">
          {navGroups.map(group => (
            <div key={group.label}>
              <div className="px-2 mb-1.5 text-[11px] font-medium text-app-text-subtle uppercase tracking-[0.06em]">{group.label}</div>
              <div className="flex flex-col gap-0.5">
                {group.items.map(item => (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    end={item.path === '/'}
                    className={({ isActive }) =>
                      cn(
                        'flex items-center gap-2.5 h-9 px-2.5 rounded-md text-[13.5px] transition-colors',
                        isActive
                          ? 'bg-app-accent-soft text-app-accent font-medium'
                          : 'text-app-text-muted hover:text-app-text hover:bg-app-surface-muted',
                      )
                    }
                  >
                    <item.icon className="w-[17px] h-[17px] shrink-0" strokeWidth={1.75} />
                    {item.label}
                  </NavLink>
                ))}
              </div>
            </div>
          ))}
        </nav>

        {/* System status */}
        <div className="m-3 p-3 rounded-lg border border-app-border bg-app-surface-muted/60">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className={cn('relative flex w-2 h-2')}>
                {status === 'connected' && <span className="absolute inset-0 rounded-full bg-app-accept opacity-40 animate-ping" />}
                <span className={cn('relative w-2 h-2 rounded-full', {
                  'bg-app-accept': status === 'connected',
                  'bg-app-warning': status === 'degraded',
                  'bg-app-reject': status === 'offline',
                })} />
              </span>
              {/* Degraded is a first-class state, not a synonym for connected:
                  a branch that could not be measured must not look like one
                  that passed. See PROJECT.md 3.7. */}
              <span className="text-[12.5px] font-medium capitalize">{status === 'offline' ? 'Backend offline' : status}</span>
            </div>
            <button onClick={toggleTheme} className="p-1 rounded text-app-text-muted hover:text-app-text hover:bg-app-surface transition-colors" title="Toggle theme">
              {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
          {/* `models` is empty whenever the heavy checkpoints are not
              installed -- the documented degraded mode. Index nothing. */}
          {health && (
            <div className="mt-2.5 flex flex-col gap-1">
              {(health.models ?? []).map(m => {
                const { role, name } = modelLabel(m);
                return (
                  <div key={m} className="flex justify-between gap-2 text-[11.5px]">
                    <span className="text-app-text-subtle">{role}</span>
                    <span className="text-app-text-muted truncate">{name}</span>
                  </div>
                );
              })}
              {!health.models?.length && <div className="text-[11.5px] text-app-warning">No models loaded</div>}
              <div className="flex justify-between gap-2 text-[11.5px]">
                <span className="text-app-text-subtle">Device</span>
                <span className="text-app-text-muted uppercase">{health.device || 'unknown'}</span>
              </div>
            </div>
          )}
        </div>
      </aside>

      <main key={location.pathname} className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
