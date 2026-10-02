import { ReactNode } from 'react';

interface PageHeaderProps {
  title: string;
  description?: ReactNode;
  eyebrow?: string;
  actions?: ReactNode;
}

export function PageHeader({ title, description, eyebrow, actions }: PageHeaderProps) {
  return (
    <header className="px-8 pt-8 pb-6 flex items-end justify-between gap-6 shrink-0">
      <div className="min-w-0 max-w-3xl">
        {eyebrow && <div className="text-[12px] font-medium text-app-text-subtle mb-1.5">{eyebrow}</div>}
        <h1 className="font-serif text-[30px] leading-[1.15] font-semibold tracking-[-0.01em] text-app-text">{title}</h1>
        {description && <p className="mt-2 text-[14px] text-app-text-muted leading-relaxed">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
    </header>
  );
}

/** The content column every page sits in. */
export function PageBody({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`px-8 pb-10 w-full max-w-[1440px] flex flex-col gap-6 animate-in ${className}`}>{children}</div>;
}
