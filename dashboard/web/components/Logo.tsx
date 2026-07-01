"use client";

export function Logo({ height = 32 }: { height?: number }) {
  const w = (height / 96) * 310;
  return (
    <svg height={height} width={w} viewBox="0 0 310 96" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="AXIOM">
      <defs>
        <linearGradient id="axiomGradMark" x1="0" y1="96" x2="96" y2="0" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#22D3EE" />
          <stop offset="1" stopColor="#34E5A1" />
        </linearGradient>
      </defs>
      <g transform="translate(7 16)">
        <path d="M8 58 36 8 64 58H46L36 38 26 58H8Z" fill="url(#axiomGradMark)" />
        <path d="M38 8 64 58H49L31 24 38 8Z" fill="#34E5A1" opacity="0.9" />
      </g>
      <text x="86" y="60" fontFamily="var(--font-sans)" fontSize="40" fontWeight="760" letterSpacing="3.5" fill="#F3F7F8">
        AXIOM
      </text>
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
      <g transform="translate(12 16)">
        <path d="M8 58 36 8 64 58H46L36 38 26 58H8Z" fill="url(#axiomGradOnly)" />
        <path d="M38 8 64 58H49L31 24 38 8Z" fill="#34E5A1" opacity="0.9" />
      </g>
    </svg>
  );
}
