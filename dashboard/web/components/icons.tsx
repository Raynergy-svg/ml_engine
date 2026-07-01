"use client";
// Shared line-icon set matching dashboard/design/target.png's icon style (thin
// stroke, currentColor, 24x24 viewBox) — replaces unicode-glyph approximations
// so the chrome is a real visual match, not a "close enough" substitute.
import type { SVGProps } from "react";

function Base(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      width="16"
      height="16"
      {...props}
    />
  );
}

export function ShieldCheckIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <path d="M12 3l7 3v5c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3z" />
      <path d="M9 12l2 2 4-4" />
    </Base>
  );
}

export function CheckCircleSolid({ size = 16, ...props }: SVGProps<SVGSVGElement> & { size?: number }) {
  return (
    <svg viewBox="0 0 20 20" width={size} height={size} {...props}>
      <circle cx="10" cy="10" r="9" fill="currentColor" />
      <path d="M6 10.2l2.6 2.6L14.2 7" stroke="var(--color-base)" strokeWidth={1.8} fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function ChevronDown(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props} width={props.width ?? 12} height={props.height ?? 12}>
      <path d="M5 8l5 5 5-5" />
    </Base>
  );
}

export function FullscreenIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <path d="M8 4H5a1 1 0 00-1 1v3M16 4h3a1 1 0 011 1v3M8 20H5a1 1 0 01-1-1v-3M16 20h3a1 1 0 001-1v-3" />
    </Base>
  );
}

export function ChartTypeIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <rect x="4" y="12" width="3" height="7" />
      <rect x="10.5" y="7" width="3" height="12" />
      <rect x="17" y="3" width="3" height="16" />
    </Base>
  );
}

export function PanelSplitIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <rect x="3.5" y="4.5" width="17" height="15" rx="1.5" />
      <path d="M14 4.5v15" strokeDasharray="2.2 2.2" />
    </Base>
  );
}

export function FitArrowsIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <path d="M9 4a5 5 0 00-5 5" />
      <path d="M15 20a5 5 0 005-5" />
      <path d="M20 4l-6 6M4 20l6-6" />
    </Base>
  );
}

export function CursorIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <path d="M5 3l5.5 15 2-6.2L19 9.5 5 3z" strokeLinejoin="round" />
    </Base>
  );
}

export function TrendlineIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <circle cx="6" cy="18" r="1.6" fill="currentColor" />
      <circle cx="18" cy="6" r="1.6" fill="currentColor" />
      <path d="M6 18L18 6" />
    </Base>
  );
}

export function PriceZoneIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <path d="M4 8h6M4 8v10M4 18h6" />
      <rect x="10" y="8" width="10" height="10" />
    </Base>
  );
}

export function BrushIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <path d="M5 19c1-3 1-6 4-6s2 3 4 3c3 0 3-4 6-9" />
      <path d="M14 4c1.5 1 2.5 2.5 3 4" />
    </Base>
  );
}

export function TextToolIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <path d="M5 6h14M12 6v13" />
    </Base>
  );
}

export function PolygonIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <path d="M12 4l7 5-2.6 8.2H7.6L5 9z" />
      <circle cx="12" cy="4" r="1.3" fill="currentColor" />
      <circle cx="19" cy="9" r="1.3" fill="currentColor" />
      <circle cx="16.4" cy="17.2" r="1.3" fill="currentColor" />
      <circle cx="7.6" cy="17.2" r="1.3" fill="currentColor" />
      <circle cx="5" cy="9" r="1.3" fill="currentColor" />
    </Base>
  );
}

export function FibIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <path d="M4 6h16M4 6a2 2 0 100 0z" />
      <circle cx="8" cy="6" r="1.3" fill="currentColor" />
      <path d="M4 12h16" />
      <circle cx="14" cy="12" r="1.3" fill="currentColor" />
      <path d="M4 18h16" />
      <circle cx="11" cy="18" r="1.3" fill="currentColor" />
    </Base>
  );
}

export function MagnetIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 6v5M12 13v0" />
      <circle cx="12" cy="12" r="1.4" fill="currentColor" />
    </Base>
  );
}

export function RulerIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <path d="M4 15l5-5 10 10-5 5z" strokeLinejoin="round" />
      <path d="M10.5 10.5l2 2M13.5 7.5l2 2M7.5 13.5l2 2" />
    </Base>
  );
}

export function CompareIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <circle cx="6" cy="6" r="2.2" />
      <circle cx="6" cy="18" r="2.2" />
      <circle cx="18" cy="12" r="2.2" />
      <path d="M8 7l8 4M8 17l8-4" />
    </Base>
  );
}

export function IndicatorsIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <rect x="4" y="4" width="16" height="16" rx="1.5" />
      <path d="M8 4v16M12 9v7" />
    </Base>
  );
}

export function SparklineIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <path d="M4 11l4 4 4-7 4 3 4-6" />
    </Base>
  );
}

export function ClockIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 7.5V12l3 2" />
    </Base>
  );
}

export function GearIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <Base {...props}>
      <circle cx="12" cy="12" r="3" />
      <path d="M12 3.5v2.2M12 18.3v2.2M20.5 12h-2.2M5.7 12H3.5M17.7 6.3l-1.5 1.5M7.8 16.2l-1.5 1.5M17.7 17.7l-1.5-1.5M7.8 7.8L6.3 6.3" />
    </Base>
  );
}
