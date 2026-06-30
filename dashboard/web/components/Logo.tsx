"use client";

export function Logo({ height = 32 }: { height?: number }) {
  const w = (height / 96) * 360;
  return (
    <svg height={height} width={w} viewBox="0 0 360 96" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="AXIOM">
      <defs>
        <linearGradient id="axiomGradMark" x1="0" y1="96" x2="96" y2="0" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#22D3EE" />
          <stop offset="1" stopColor="#34E5A1" />
        </linearGradient>
      </defs>
      <g transform="translate(8 8)">
        <path d="M10 72 L40 12" stroke="url(#axiomGradMark)" strokeWidth="7" strokeLinecap="round" />
        <path d="M40 12 L70 72" stroke="url(#axiomGradMark)" strokeWidth="7" strokeLinecap="round" />
        <path d="M40 12 L52 4" stroke="url(#axiomGradMark)" strokeWidth="7" strokeLinecap="round" />
        <path d="M22 48 L58 48" stroke="#E6EDF3" strokeWidth="5" strokeLinecap="round" opacity="0.92" />
        <circle cx="22" cy="48" r="3.4" fill="#22D3EE" />
        <circle cx="58" cy="48" r="3.4" fill="#34E5A1" />
      </g>
      <text x="104" y="60" fontFamily="var(--font-sans)" fontSize="40" fontWeight="650" letterSpacing="6" fill="#E6EDF3">
        AXIOM
      </text>
      <rect x="105" y="72" width="208" height="2.5" rx="1.25" fill="url(#axiomGradMark)" opacity="0.85" />
    </svg>
  );
}

/** Compact mark only (no wordmark) for tight spaces. */
export function LogoMark({ size = 28 }: { size?: number }) {
  return (
    <svg height={size} width={size} viewBox="0 0 96 96" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="AXIOM">
      <defs>
        <linearGradient id="axiomGradOnly" x1="0" y1="96" x2="96" y2="0" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#22D3EE" />
          <stop offset="1" stopColor="#34E5A1" />
        </linearGradient>
      </defs>
      <g transform="translate(8 8)">
        <path d="M10 72 L40 12" stroke="url(#axiomGradOnly)" strokeWidth="7" strokeLinecap="round" />
        <path d="M40 12 L70 72" stroke="url(#axiomGradOnly)" strokeWidth="7" strokeLinecap="round" />
        <path d="M40 12 L52 4" stroke="url(#axiomGradOnly)" strokeWidth="7" strokeLinecap="round" />
        <path d="M22 48 L58 48" stroke="#E6EDF3" strokeWidth="5" strokeLinecap="round" opacity="0.92" />
        <circle cx="22" cy="48" r="3.4" fill="#22D3EE" />
        <circle cx="58" cy="48" r="3.4" fill="#34E5A1" />
      </g>
    </svg>
  );
}
