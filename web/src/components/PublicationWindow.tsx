import { CalendarDays, ChevronLeft, ChevronRight } from "lucide-react";
import { useState } from "react";
import { MONTHS, monthKey, publicationLabel, type PublicationWindow } from "../publication";

function YearSelect({ label, value, onChange }: { label: string; value: number; onChange: (year: number) => void }) {
  const [yearLimit] = useState(() => new Date().getUTCFullYear());
  return <label>{label}<select value={value} onChange={(event) => onChange(Number(event.target.value))}>
    {Array.from({ length: yearLimit - 1990 }, (_, index) => yearLimit - index).map((year) => <option key={year} value={year}>{year}</option>)}
  </select></label>;
}

function MonthSelect({ label, value, onChange }: { label: string; value: string; onChange: (month: string) => void }) {
  const [year, month] = value.split("-").map(Number);
  return <fieldset className="month-pair"><legend>{label}</legend><div>
    <YearSelect label={`${label} year`} value={year!} onChange={(next) => onChange(monthKey(next, month!))} />
    <label>{label} month<select value={month} onChange={(event) => onChange(monthKey(year!, Number(event.target.value)))}>
      {MONTHS.map((name, index) => <option key={name} value={index + 1}>{name}</option>)}
    </select></label>
  </div></fieldset>;
}

/** Share explicit UTC publication windows between library filtering and source sync. */
export function PublicationWindowFields({ value, onChange, allowAll = false }: {
  value: PublicationWindow; onChange: (window: PublicationWindow) => void; allowAll?: boolean;
}) {
  const [now] = useState(() => new Date());
  const yearLimit = now.getUTCFullYear();
  const thisMonth = monthKey(now.getUTCFullYear(), now.getUTCMonth() + 1);
  const update = (change: Partial<PublicationWindow>) => onChange({ ...value, ...change });
  const changeMode = (mode: PublicationWindow["mode"]) => onChange({
    mode, initial_months: value.initial_months,
    ...(["year", "month"].includes(mode) ? { year: now.getUTCFullYear() } : {}),
    ...(mode === "month" ? { month: now.getUTCMonth() + 1 } : {}),
    ...(["since", "range"].includes(mode) ? { start_month: thisMonth } : {}),
    ...(mode === "range" ? { end_month: thisMonth } : {}),
  });
  return <div className="publication-window">
    <label>Publication window<select value={value.mode} onChange={(event) => changeMode(event.target.value as PublicationWindow["mode"])}>
      <option value="relative">Recent months</option><option value="year">Whole year</option>
      <option value="month">One month in a year</option><option value="since">From a month onward</option>
      <option value="range">Month range</option>{allowAll && <option value="all">All dates</option>}
    </select></label>
    {(value.mode === "relative" || (allowAll && value.mode === "all")) && <label>Number of months<select value={value.mode === "all" ? "all" : value.initial_months} onChange={(event) => {
      if (event.target.value === "all") changeMode("all");
      else onChange({ mode: "relative", initial_months: Number(event.target.value) });
    }}>
      {[...new Set([1, 3, 6, 12, 24, value.initial_months])].sort((a, b) => a - b).map((months) => <option key={months} value={months}>Last {months} month{months === 1 ? "" : "s"}</option>)}
      {allowAll && <option value="all">All dates</option>}
    </select></label>}
    {["year", "month"].includes(value.mode) && <YearSelect label="Publication year" value={value.year ?? now.getUTCFullYear()} onChange={(year) => update({ year })} />}
    {value.mode === "month" && <fieldset className="month-picker"><legend>Publication month</legend>
      <div className="month-picker-header"><CalendarDays size={14} /><span>{publicationLabel(value)}</span>
        <button type="button" aria-label="Previous year" disabled={value.year === 1991} onClick={() => update({ year: (value.year ?? yearLimit) - 1 })}><ChevronLeft size={14} /></button>
        <button type="button" aria-label="Next year" disabled={(value.year ?? yearLimit) >= yearLimit} onClick={() => update({ year: (value.year ?? yearLimit) + 1 })}><ChevronRight size={14} /></button>
      </div><div className="month-grid">{MONTHS.map((name, index) => <button type="button" key={name} aria-label={name} aria-pressed={value.month === index + 1} onClick={() => update({ month: index + 1 })}>{name.slice(0, 3)}</button>)}</div>
    </fieldset>}
    {["since", "range"].includes(value.mode) && <MonthSelect label="From month" value={value.start_month ?? thisMonth} onChange={(start_month) => update({ start_month, ...(value.end_month && value.end_month < start_month ? { end_month: start_month } : {}) })} />}
    {value.mode === "range" && <MonthSelect label="Through month" value={value.end_month ?? thisMonth} onChange={(end_month) => update({ end_month, ...(value.start_month && value.start_month > end_month ? { start_month: end_month } : {}) })} />}
    <p className="publication-hint">Publication dates · UTC · {value.mode === "all" ? "all saved papers" : value.mode === "relative" ? "rolling window" : "full months included"}</p>
  </div>;
}
