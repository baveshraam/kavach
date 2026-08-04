import { ReactNode } from 'react';

interface PageHeaderProps {
  title: string;
  actions?: ReactNode;
}

export function PageHeader({ title, actions }: PageHeaderProps) {
  return (
    <div className="h-[48px] px-6 border-b border-app-border flex items-center justify-between shrink-0 bg-app-surface">
      <h2 className="text-[15px] font-semibold">{title}</h2>
      {actions && <div className="flex gap-4">{actions}</div>}
    </div>
  );
}
