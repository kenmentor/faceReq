"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navLinks = [
  { href: "/", label: "Home" },
  { href: "/enroll", label: "Enroll" },
  { href: "/verify", label: "Verify" },
  { href: "/history", label: "History" },
  { href: "/settings", label: "Settings" },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav style={{ borderBottom: '1px solid var(--border)' }} className="sticky top-0 z-50 bg-black">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center justify-between h-14">
          <Link href="/" className="flex items-center gap-3">
            <span className="text-2xl">👁️</span>
            <span style={{ color: 'var(--text-primary)' }} className="text-xl font-bold">
              FaceID
            </span>
          </Link>

          <div className="hidden md:flex items-center gap-8">
            {navLinks.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  style={{
                    color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                    borderBottom: isActive ? '2px solid var(--accent)' : '2px solid transparent',
                    paddingBottom: '2px'
                  }}
                  className="text-sm font-medium transition-colors hover:text-white"
                >
                  {link.label}
                </Link>
              );
            })}
          </div>

          <div className="md:hidden">
            <select
              value={pathname}
              onChange={(e) => {
                window.location.href = e.target.value;
              }}
              style={{ background: 'var(--surface)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
              className="px-3 py-1.5 rounded-lg text-sm appearance-none cursor-pointer"
            >
              {navLinks.map((link) => (
                <option key={link.href} value={link.href}>
                  {link.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
    </nav>
  );
}
