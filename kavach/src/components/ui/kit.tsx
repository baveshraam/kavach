/**
 * The component kit every page is built from.
 *
 * One place owns spacing, radius, type scale and colour roles, so a page
 * cannot drift into its own idiom. The rules, in short:
 *   - 4px spacing grid; cards pad 20px, dense tables 12px.
 *   - Radius 8px on surfaces, 6px on controls. Nothing is pill-shaped except badges.
 *   - Type: 12 / 13 / 14 / 16 / 20 / 28. Labels are 12px medium, never all-caps
 *     monospace walls; mono is for numbers and identifiers only.
 *   - Status colour only on status. Tamil / English colour only on language.
 */
import { Fragment, ReactNode, ButtonHTMLAttributes, InputHTMLAttributes, SelectHTMLAttributes, forwardRef } from 'react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { AlertTriangle, Info, CheckCircle2, XCircle, Loader2 } from 'lucide-react';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// ------------------------------------------------------------------ surfaces

export function Card({ className, children, ...rest }: { className?: string; children: ReactNode } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('bg-app-surface border border-app-border rounded-lg', className)} {...rest}>
      {children}
    </div>
  );
}

export function CardHeader({ title, subtitle, actions, className }: { title: ReactNode; subtitle?: ReactNode; actions?: ReactNode; className?: string }) {
  return (
    <div className={cn('flex items-start justify-between gap-4 px-5 pt-4 pb-3', className)}>
      <div className="min-w-0">
        <h3 className="text-[14px] font-semibold text-app-text leading-snug">{title}</h3>
        {subtitle && <p className="text-[12.5px] text-app-text-muted mt-0.5 leading-snug">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
    </div>
  );
}

export function CardBody({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn('px-5 pb-5', className)}>{children}</div>;
}

// ------------------------------------------------------------------- buttons

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
type ButtonSize = 'sm' | 'md' | 'lg';

const buttonVariants: Record<ButtonVariant, string> = {
  primary: 'bg-app-accent text-app-on-accent hover:bg-app-accent-hover border border-transparent',
  secondary: 'bg-app-surface text-app-text border border-app-border-strong hover:bg-app-surface-muted',
  ghost: 'text-app-text-muted hover:text-app-text hover:bg-app-surface-muted border border-transparent',
  danger: 'text-app-reject border border-app-border hover:bg-app-reject hover:text-white hover:border-app-reject',
};

const buttonSizes: Record<ButtonSize, string> = {
  sm: 'h-8 px-3 text-[12.5px] gap-1.5',
  md: 'h-9 px-4 text-[13px] gap-2',
  lg: 'h-11 px-5 text-[14px] gap-2',
};

export const Button = forwardRef<HTMLButtonElement, ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; size?: ButtonSize; loading?: boolean; icon?: ReactNode }>(
  function Button({ variant = 'secondary', size = 'md', loading, icon, className, children, disabled, ...rest }, ref) {
    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={cn(
          'inline-flex items-center justify-center rounded-md font-medium transition-colors duration-150 select-none whitespace-nowrap',
          'disabled:opacity-45 disabled:pointer-events-none',
          buttonVariants[variant],
          buttonSizes[size],
          className,
        )}
        {...rest}
      >
        {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : icon}
        {children}
      </button>
    );
  },
);

// -------------------------------------------------------------------- inputs

export function Field({ label, hint, children, className }: { label: ReactNode; hint?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <label className={cn('flex flex-col gap-1.5 min-w-0', className)}>
      <span className="text-[12px] font-medium text-app-text-muted">{label}</span>
      {children}
      {hint && <span className="text-[11.5px] text-app-text-subtle">{hint}</span>}
    </label>
  );
}

const controlBase =
  'h-9 w-full rounded-md border border-app-border-strong bg-app-surface px-3 text-[13px] text-app-text ' +
  'placeholder:text-app-text-subtle transition-colors focus:outline-none focus:border-app-accent focus:ring-2 focus:ring-app-accent/15 ' +
  'disabled:bg-app-surface-muted disabled:text-app-text-subtle';

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input({ className, ...rest }, ref) {
  return <input ref={ref} className={cn(controlBase, className)} {...rest} />;
});

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(function Select({ className, children, ...rest }, ref) {
  return (
    <select ref={ref} className={cn(controlBase, 'cursor-pointer', className)} {...rest}>
      {children}
    </select>
  );
});

export function Segmented<T extends string>({ value, options, onChange, size = 'md' }: { value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void; size?: 'sm' | 'md' }) {
  return (
    <div className="inline-flex p-0.5 rounded-md bg-app-surface-muted border border-app-border">
      {options.map(o => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={cn(
            'rounded-[5px] font-medium transition-colors',
            size === 'sm' ? 'h-7 px-2.5 text-[12px]' : 'h-8 px-3 text-[12.5px]',
            value === o.value ? 'bg-app-surface text-app-text shadow-[0_1px_2px_rgba(0,0,0,0.06)] border border-app-border' : 'text-app-text-muted hover:text-app-text border border-transparent',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// -------------------------------------------------------------------- badges

export type Tone = 'neutral' | 'accent' | 'accept' | 'reject' | 'warning' | 'ta' | 'en';

const toneClasses: Record<Tone, string> = {
  neutral: 'bg-app-surface-muted text-app-text-muted border-app-border',
  accent: 'bg-app-accent-soft text-app-accent border-transparent',
  accept: 'bg-app-accept-soft text-app-accept border-transparent',
  reject: 'bg-app-reject-soft text-app-reject border-transparent',
  warning: 'bg-app-warning-soft text-app-warning border-transparent',
  ta: 'bg-app-ta-soft text-app-ta border-transparent',
  en: 'bg-app-en-soft text-app-en border-transparent',
};

export function Badge({ tone = 'neutral', children, className, dot }: { tone?: Tone; children: ReactNode; className?: string; dot?: boolean }) {
  return (
    <span className={cn('inline-flex items-center gap-1.5 h-[22px] px-2 rounded-full border text-[11.5px] font-medium whitespace-nowrap', toneClasses[tone], className)}>
      {dot && <span className="w-1.5 h-1.5 rounded-full bg-current" />}
      {children}
    </span>
  );
}

export function DecisionBadge({ decision, size = 'sm' }: { decision: 'ACCEPT' | 'REJECT' | 'BORDERLINE'; size?: 'sm' | 'lg' }) {
  const tone: Tone = decision === 'ACCEPT' ? 'accept' : decision === 'REJECT' ? 'reject' : 'warning';
  const label = decision === 'ACCEPT' ? 'Accepted' : decision === 'REJECT' ? 'Rejected' : 'Borderline';
  if (size === 'lg') {
    const Icon = decision === 'ACCEPT' ? CheckCircle2 : decision === 'REJECT' ? XCircle : AlertTriangle;
    return (
      <span className={cn('inline-flex items-center gap-2 h-10 px-4 rounded-md text-[16px] font-semibold', toneClasses[tone])}>
        <Icon className="w-5 h-5" /> {label}
      </span>
    );
  }
  return <Badge tone={tone} dot>{label}</Badge>;
}

// -------------------------------------------------------------------- notice

export function Notice({ tone = 'info', title, children, className }: { tone?: 'info' | 'warning' | 'reject' | 'accept'; title?: ReactNode; children?: ReactNode; className?: string }) {
  const styles = {
    info: 'bg-app-accent-soft/60 border-app-accent/20 text-app-text',
    warning: 'bg-app-warning-soft border-app-warning/25 text-app-text',
    reject: 'bg-app-reject-soft border-app-reject/25 text-app-text',
    accept: 'bg-app-accept-soft border-app-accept/25 text-app-text',
  }[tone];
  const iconColor = { info: 'text-app-accent', warning: 'text-app-warning', reject: 'text-app-reject', accept: 'text-app-accept' }[tone];
  const Icon = tone === 'info' ? Info : tone === 'accept' ? CheckCircle2 : AlertTriangle;
  return (
    <div className={cn('flex gap-3 rounded-lg border px-4 py-3 text-[13px] leading-relaxed', styles, className)}>
      <Icon className={cn('w-4 h-4 mt-[3px] shrink-0', iconColor)} />
      <div className="min-w-0">
        {title && <div className="font-semibold mb-0.5">{title}</div>}
        {children && <div className="text-app-text-muted">{children}</div>}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------- stats

export function Stat({ label, value, hint, hintTone }: { label: ReactNode; value: ReactNode; hint?: ReactNode; hintTone?: 'muted' | 'warning' | 'accept' | 'reject' }) {
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <span className="text-[12px] font-medium text-app-text-muted truncate">{label}</span>
      <span className="text-[26px] leading-none font-semibold tracking-tight tnum text-app-text">{value}</span>
      {hint && (
        <span className={cn('text-[12px] truncate', {
          'text-app-text-subtle': !hintTone || hintTone === 'muted',
          'text-app-warning': hintTone === 'warning',
          'text-app-accept': hintTone === 'accept',
          'text-app-reject': hintTone === 'reject',
        })}>{hint}</span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- empty/load

export function EmptyState({ icon, title, children, action }: { icon?: ReactNode; title: ReactNode; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-12 px-6 gap-2">
      {icon && <div className="w-10 h-10 rounded-full bg-app-surface-muted flex items-center justify-center text-app-text-subtle mb-1">{icon}</div>}
      <div className="text-[14px] font-medium text-app-text">{title}</div>
      {children && <div className="text-[13px] text-app-text-muted max-w-sm leading-relaxed">{children}</div>}
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-[13px] text-app-text-muted">
      <Loader2 className="w-4 h-4 animate-spin" /> {label}
    </div>
  );
}

// -------------------------------------------------------------------- tables

export function Table({ children, className }: { children: ReactNode; className?: string }) {
  return <table className={cn('w-full text-left border-collapse text-[13px]', className)}>{children}</table>;
}

export function Th({ children, className, align = 'left' }: { children?: ReactNode; className?: string; align?: 'left' | 'right' | 'center' }) {
  return (
    <th className={cn('px-4 h-9 text-[12px] font-medium text-app-text-muted border-b border-app-border bg-app-surface-muted/60 whitespace-nowrap',
      align === 'right' && 'text-right', align === 'center' && 'text-center', className)}>
      {children}
    </th>
  );
}

export function Td({ children, className, align = 'left', mono, ...rest }: { children?: ReactNode; className?: string; align?: 'left' | 'right' | 'center'; mono?: boolean } & React.TdHTMLAttributes<HTMLTableCellElement>) {
  return (
    <td className={cn('px-4 py-2.5 border-b border-app-border align-middle',
      align === 'right' && 'text-right', align === 'center' && 'text-center', mono && 'mono text-[12.5px]', className)} {...rest}>
      {children}
    </td>
  );
}

// ------------------------------------------------------------------ language

/** The Tamil / English split as a two-colour bar. The one chart shape the whole UI shares. */
export function LangBar({ ta, en, neutral = 0, className, showLabels }: { ta: number; en: number; neutral?: number; className?: string; showLabels?: boolean }) {
  const total = ta + en + neutral || 1;
  const pTa = (ta / total) * 100;
  const pEn = (en / total) * 100;
  return (
    <div className={cn('flex flex-col gap-1', className)}>
      <div className="flex h-2 w-full rounded-full overflow-hidden bg-app-surface-muted">
        <div className="bg-app-ta" style={{ width: `${pTa}%` }} title={`Tamil ${pTa.toFixed(0)}%`} />
        <div className="bg-app-en" style={{ width: `${pEn}%` }} title={`English ${pEn.toFixed(0)}%`} />
      </div>
      {showLabels && (
        <div className="flex justify-between text-[11.5px] tnum">
          <span className="text-app-ta font-medium">TA {pTa.toFixed(0)}%</span>
          <span className="text-app-en font-medium">EN {pEn.toFixed(0)}%</span>
        </div>
      )}
    </div>
  );
}

export function LangLegend({ className }: { className?: string }) {
  return (
    <div className={cn('flex items-center gap-4 text-[12px] text-app-text-muted', className)}>
      <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-app-ta" /> Tamil</span>
      <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-app-en" /> English</span>
      <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-app-text-subtle" /> Neutral / name</span>
    </div>
  );
}

/** A transcript with each token coloured by its language. Hover shows class and confidence. */
export function TokenText({ tokens, className }: { tokens: { text: string; language: string; semanticClass: string; lidConfidence: number }[]; className?: string }) {
  return (
    <p className={cn('leading-[1.9] text-[15px]', className)}>
      {tokens.map((t, i) => (
        // A real space between spans, not a CSS margin: the margin looked right
        // but left the DOM text with no whitespace, so a copied transcript, a
        // screen reader and find-in-page all saw one run-together word.
        <Fragment key={i}>
          {i > 0 && ' '}
          <span
            title={`${t.language} · ${t.semanticClass.replace(/_/g, ' ').toLowerCase()} · ${(t.lidConfidence * 100).toFixed(0)}%`}
            className={cn('rounded-[3px] px-[2px] -mx-[2px] cursor-help transition-colors hover:bg-app-surface-muted',
              t.language === 'TA' && 'text-app-ta',
              t.language === 'EN' && 'text-app-en',
              t.language === 'NAMED_ENTITY' && 'underline decoration-dotted underline-offset-4 text-app-text',
              t.language === 'NEUTRAL' && 'text-app-text-subtle',
            )}
          >
            {t.text}
          </span>
        </Fragment>
      ))}
    </p>
  );
}

// ---------------------------------------------------------------- formatting

export const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined || !Number.isFinite(v) ? 'n/a' : `${(v * 100).toFixed(digits)}%`;

export const minutes = (sec: number) => (sec < 3600 ? `${(sec / 60).toFixed(1)} min` : `${(sec / 3600).toFixed(2)} h`);

export const titleCase = (s: string) => s.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase());

export const branchLabel: Record<string, string> = {
  speaker_embedding: 'Voiceprint (ECAPA)',
  csbg: 'Code-switch graph',
  knowledge: 'Knowledge answer',
  liveness: 'Liveness',
  signal_integrity: 'Signal integrity',
};
