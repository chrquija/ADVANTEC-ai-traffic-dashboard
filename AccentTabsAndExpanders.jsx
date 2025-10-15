// AccentTabsAndExpanders.jsx
// Self-contained React + Tailwind component: blue-family tabs with matching expanders
// A11y + keyboard support; purge-safe Tailwind classes via explicit maps

import React, { useId, useMemo, useState, useCallback } from "react";

/** ------------------------------------------------------------------
 * CONFIG: Add/edit pages here. Each has a title, key, and color token
 * (must exist in COLOR_MAP below to stay purge-safe for Tailwind).
 * ------------------------------------------------------------------ */
const PAGES = [
  { key: "pg1", title: "Pg.1 ITERIS CLEARGUIDE SETTINGS", color: "blue-400", subtitle: "Travel-time APIs & sources" },
  { key: "pg2", title: "Pg.2 KINETIC MOBILITY SETTINGS", color: "sky-400", subtitle: "Corridor mobility configuration" },
  { key: "pg3", title: "Pg.3 ACYCLICA SETTINGS", color: "indigo-400", subtitle: "Signal analytics & data feeds" },
  { key: "pg4", title: "Pg.4 ITERIS VANTAGE LIVE SETTINGS", color: "cyan-400", subtitle: "CCTV & sensor stream setup" },
  { key: "pg5", title: "Pg.5 BOSCH SETTINGS", color: "blue-300", subtitle: "Edge devices & credentials" },
];

/** ------------------------------------------------------------------
 * Purge-safe Tailwind class maps for borders, bg bars, and icon tints.
 * DO NOT build class strings dynamically; use these maps instead.
 * ------------------------------------------------------------------ */
const COLOR_MAP = {
  "blue-400": {
    border: "border-blue-400",
    borderThick: "border-l-[6px] border-blue-400",
    bar: "bg-blue-400",
    barSoft20: "bg-blue-400/20",
    barSoft60: "bg-blue-400/60",
    barSoft60Hover: "group-hover:bg-blue-400/60",
    icon: "text-blue-300",
  },
  "sky-400": {
    border: "border-sky-400",
    borderThick: "border-l-[6px] border-sky-400",
    bar: "bg-sky-400",
    barSoft20: "bg-sky-400/20",
    barSoft60: "bg-sky-400/60",
    barSoft60Hover: "group-hover:bg-sky-400/60",
    icon: "text-sky-300",
  },
  "indigo-400": {
    border: "border-indigo-400",
    borderThick: "border-l-[6px] border-indigo-400",
    bar: "bg-indigo-400",
    barSoft20: "bg-indigo-400/20",
    barSoft60: "bg-indigo-400/60",
    barSoft60Hover: "group-hover:bg-indigo-400/60",
    icon: "text-indigo-300",
  },
  "cyan-400": {
    border: "border-cyan-400",
    borderThick: "border-l-[6px] border-cyan-400",
    bar: "bg-cyan-400",
    barSoft20: "bg-cyan-400/20",
    barSoft60: "bg-cyan-400/60",
    barSoft60Hover: "group-hover:bg-cyan-400/60",
    icon: "text-cyan-300",
  },
  "blue-300": {
    border: "border-blue-300",
    borderThick: "border-l-[6px] border-blue-300",
    bar: "bg-blue-300",
    barSoft20: "bg-blue-300/20",
    barSoft60: "bg-blue-300/60",
    barSoft60Hover: "group-hover:bg-blue-300/60",
    icon: "text-blue-200",
  },
};

/** Simple chevron icon */
const Chevron = ({ className = "" }) => (
  <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
    <path d="M6 9l6 6 6-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

/** Optional: page icon (muted) */
const DotIcon = ({ className = "" }) => (
  <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
    <circle cx="12" cy="12" r="5" fill="currentColor" />
  </svg>
);

/** Tabs with subtle bottom accent bars (matching page color) */
function AccentTabs({ pages, activeKey, onChange }) {
  const listId = useId();

  // Keyboard navigation (Left/Right, Enter/Space activates)
  const onKeyDown = useCallback((e) => {
    const idx = pages.findIndex(p => p.key === activeKey);
    if (idx === -1) return;
    if (e.key === "ArrowRight") {
      e.preventDefault();
      const next = (idx + 1) % pages.length;
      onChange(pages[next].key);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      const prev = (idx - 1 + pages.length) % pages.length;
      onChange(pages[prev].key);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      // no-op here, click is handled on buttons
    }
  }, [pages, activeKey, onChange]);

  return (
    <nav
      id={listId}
      role="tablist"
      aria-label="Select Page"
      className="flex flex-wrap gap-2"
      onKeyDown={onKeyDown}
    >
      {pages.map((p) => {
        const cm = COLOR_MAP[p.color] || COLOR_MAP["blue-400"];
        const selected = p.key === activeKey;
        const tabId = `tab-${p.key}`;
        const panelId = `panel-${p.key}`;
        return (
          <button
            key={p.key}
            id={tabId}
            role="tab"
            aria-selected={selected}
            aria-controls={panelId}
            data-active={selected ? "true" : "false"}
            onClick={() => onChange(p.key)}
            className={[
              "group relative px-3 py-2 text-sm rounded-md",
              "text-slate-200 hover:text-white transition-colors",
              "focus:outline-none focus:ring-2 focus:ring-cyan-400/50",
              selected ? "font-semibold" : "font-medium"
            ].join(" ")}
          >
            {p.title.replace(/ SETTINGS.*/i, "")}
            {/* bottom accent bar */}
            <span
              aria-hidden="true"
              className={[
                "pointer-events-none absolute left-0 right-0 -bottom-0.5 h-[2px]",
                cm.barSoft20,
                cm.barSoft60Hover,
                selected ? cm.bar : ""
              ].join(" ")}
            />
          </button>
        );
      })}
    </nav>
  );
}

/** Single expander item */
function ExpanderItem({ page, open, onToggle }) {
  const cm = COLOR_MAP[page.color] || COLOR_MAP["blue-400"];
  const panelId = `panel-${page.key}`;
  const headerId = `header-${page.key}`;

  return (
    <div className="rounded-2xl bg-slate-800 shadow-md hover:shadow-lg transition-shadow">
      <button
        id={headerId}
        aria-expanded={open}
        aria-controls={panelId}
        onClick={onToggle}
        className={[
          "group w-full flex items-center justify-between",
          "rounded-2xl bg-slate-800 hover:bg-slate-700",
          "transition-all pl-3 pr-3 py-3",
          "focus:outline-none focus:ring-2 focus:ring-cyan-400/50",
          open ? cm.borderThick : `border-l-4 ${cm.border}`,
        ].join(" ")}
      >
        <div className="flex items-center gap-3">
          <DotIcon className={["w-5 h-5", (COLOR_MAP[page.color]?.icon || "text-blue-300")].join(" ")} />
          <div>
            <div className="text-sm font-semibold text-slate-200 group-hover:text-white">
              {page.title}
            </div>
            {page.subtitle && (
              <div className="text-xs text-slate-400">{page.subtitle}</div>
            )}
          </div>
        </div>
        <Chevron
          className={[
            "w-4 h-4 text-slate-300 transition-transform duration-200",
            open ? "rotate-180" : ""
          ].join(" ")}
        />
      </button>

      {/* Panel */}
      <div
        id={panelId}
        role="tabpanel"
        aria-labelledby={headerId}
        className={[
          "overflow-hidden transition-[max-height,opacity] duration-200",
          open ? "max-h-96 opacity-100" : "max-h-0 opacity-0"
        ].join(" ")}
      >
        <div className="px-4 pb-4 pt-2 text-sm text-slate-200">
          {/* Placeholder content block — replace with real settings */}
          <div className="rounded-xl border border-slate-700/60 bg-slate-900/40 p-4">
            <p className="mb-2 font-medium">Settings for: {page.title}</p>
            <ul className="list-disc pl-5 space-y-1 text-slate-300">
              <li>Input fields, toggles, selectors go here.</li>
              <li>Keep content compact; use grid for forms.</li>
              <li>Maintain dark theme and a11y contrast.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Expanders list with open/close state per page */
function AccentExpanders({ pages, activeKey }) {
  const [openKey, setOpenKey] = useState(activeKey);

  // Sync the expander to the active tab
  React.useEffect(() => {
    setOpenKey(activeKey);
  }, [activeKey]);

  return (
    <div className="space-y-2 mt-3">
      {pages.map((p) => (
        <ExpanderItem
          key={p.key}
          page={p}
          open={openKey === p.key}
          onToggle={() => setOpenKey(openKey === p.key ? null : p.key)}
        />
      ))}
    </div>
  );
}

/** Main exported component: Tabs + Expanders (linked by key) */
export default function AccentTabsAndExpanders() {
  const [active, setActive] = useState(PAGES[0].key);
  const pages = useMemo(() => PAGES, []);

  return (
    <section className="w-full">
      {/* Tabs */}
      <AccentTabs pages={pages} activeKey={active} onChange={setActive} />

      {/* Expanders */}
      <AccentExpanders pages={pages} activeKey={active} />
    </section>
  );
}
