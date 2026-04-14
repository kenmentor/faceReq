import Link from "next/link";

export default function Home() {
  return (
    <div style={{ background: 'var(--background)', minHeight: 'calc(100vh - 57px)' }}>
      <div className="max-w-lg mx-auto px-4 py-12">
        <div className="text-center mb-10">
          <h1 style={{ color: 'var(--text-primary)' }} className="text-4xl font-bold mb-3">
            Face Recognition
          </h1>
          <p style={{ color: 'var(--text-secondary)' }} className="text-base">
            Multi-Model Face Verification System
          </p>
        </div>

        <div className="space-y-3">
          <NavCard href="/enroll" title="Enroll" desc="Register new users" />
          <NavCard href="/verify" title="Recognize" desc="Recognize face identity" />
          <NavCard href="/history" title="History" desc="View verification logs" />
          <NavCard href="/settings" title="Settings" desc="Configure preferences" />
        </div>
      </div>
    </div>
  );
}

function NavCard({ href, title, desc }: { href: string; title: string; desc: string }) {
  return (
    <Link
      href={href}
      style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        display: 'block',
        padding: '20px 24px',
        borderRadius: '12px',
        transition: 'background 0.2s'
      }}
      className="hover:bg-[var(--surface-hover)]"
    >
      <div className="flex items-center justify-between">
        <div>
          <h3 style={{ color: 'var(--text-primary)' }} className="text-lg font-semibold">{title}</h3>
          <p style={{ color: 'var(--text-secondary)' }} className="text-sm mt-1">{desc}</p>
        </div>
        <svg style={{ color: 'var(--text-secondary)' }} width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M9 18l6-6-6-6" />
        </svg>
      </div>
    </Link>
  );
}
