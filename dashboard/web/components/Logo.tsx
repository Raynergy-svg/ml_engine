"use client";

export function Logo({ height = 32 }: { height?: number }) {
  const w = (height / 48) * 214;
  return (
    <svg height={height} width={w} viewBox="0 0 214 48" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="AXIOM">
      <g transform="translate(2 7)" fill="#2FE084">
        <path d="M0.5 32 14.8 3.6 23.5 3.6 10.7 32H0.5Z" />
        <path d="M18.2 3.6 35.2 32H24.2L12.5 11.4 18.2 3.6Z" />
      </g>
      <text
        x="48"
        y="34"
        fontFamily="Inter, var(--font-sans), Arial, sans-serif"
        fontSize="28"
        fontWeight="800"
        letterSpacing="1.15"
        fill="#F4F7F8"
      >
        AXIOM
      </text>
    </svg>
  );
}

/** Compact mark only (no wordmark) for tight spaces. */
export function LogoMark({ size = 28 }: { size?: number }) {
  return (
    <svg height={size} width={size} viewBox="0 0 96 96" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="AXIOM">
      <g transform="translate(13 14) scale(1.95)" fill="#2FE084">
        <path d="M0.5 32 14.8 3.6 23.5 3.6 10.7 32H0.5Z" />
        <path d="M18.2 3.6 35.2 32H24.2L12.5 11.4 18.2 3.6Z" />
      </g>
    </svg>
  );
}
