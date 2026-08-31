import React, { useMemo, useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import {
  CalendarDays,
  Download,
  LineChart as LineChartIcon,
} from "lucide-react";


function formatPrice(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "--";
  }

  return number.toLocaleString("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  });
}


function formatDate(value) {
  if (!value) {
    return "--";
  }

  const date = new Date(`${value}T00:00:00`);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
  });
}


function PredictionTooltip({
  active,
  payload,
  label,
}) {
  if (!active || !payload?.length) {
    return null;
  }

  const row = payload[0]?.payload || {};

  return (
    <div className="min-w-[220px] rounded-xl border border-white/10 bg-[#0b1018] p-3 shadow-2xl">
      <p className="text-[11px] font-semibold text-white">
        {label}
      </p>

      <div className="mt-3 space-y-2 text-[10px]">
        <div className="flex items-center justify-between gap-5">
          <span className="text-blue-400">
            X2 Predicted
          </span>
          <strong className="text-white">
            {formatPrice(row.x2Point)}
          </strong>
        </div>

        <div className="flex items-center justify-between gap-5">
          <span className="text-green-400">
            Actual Close
          </span>
          <strong className="text-white">
            {formatPrice(row.actualClose)}
          </strong>
        </div>

        <div className="flex items-center justify-between gap-5">
          <span className="text-gray-500">
            80% Range Low
          </span>
          <strong className="text-gray-300">
            {formatPrice(row.rangeLow)}
          </strong>
        </div>

        <div className="flex items-center justify-between gap-5">
          <span className="text-gray-500">
            80% Range High
          </span>
          <strong className="text-gray-300">
            {formatPrice(row.rangeHigh)}
          </strong>
        </div>

        {Number.isFinite(row.absError) && (
          <div className="flex items-center justify-between gap-5 border-t border-white/5 pt-2">
            <span className="text-gray-500">
              Abs Error
            </span>
            <strong className="text-amber-300">
              {formatPrice(row.absError)}
            </strong>
          </div>
        )}
      </div>
    </div>
  );
}


export default function PredictionHistoryChart({
  history = [],
  symbol = "RELIANCE.NS",
}) {
  const [recordLimit, setRecordLimit] =
    useState(20);


  const chartData = useMemo(() => {
    return [...history]
      .slice(0, recordLimit)
      .map((row) => {
        const rangeLow =
          Number(row.expected_range_lower);

        const rangeHigh =
          Number(row.expected_range_upper);

        const x2Point =
          Number(
            row.experimental_x2_point_price ??
              row.predicted_price
          );

        const actualClose =
          Number(row.actual_close);

        const absError =
          Number(
            row.experimental_x2_point_absolute_error ??
              row.absolute_error
          );

        return {
          date: formatDate(
            row.target_date || row.base_date
          ),

          fullDate:
            row.target_date ||
            row.base_date ||
            "--",

          x2Point:
            Number.isFinite(x2Point)
              ? x2Point
              : null,

          actualClose:
            Number.isFinite(actualClose)
              ? actualClose
              : null,

          rangeLow:
            Number.isFinite(rangeLow)
              ? rangeLow
              : null,

          rangeHigh:
            Number.isFinite(rangeHigh)
              ? rangeHigh
              : null,

          rangeBand:
            Number.isFinite(rangeLow) &&
            Number.isFinite(rangeHigh)
              ? Math.max(
                  0,
                  rangeHigh - rangeLow
                )
              : null,

          absError:
            Number.isFinite(absError)
              ? absError
              : null,

          status:
            row.status || "PENDING",
        };
      })
      .reverse();
  }, [history, recordLimit]);


  const exportCsv = () => {
    if (!history.length) {
      return;
    }

    const headers = [
      "Base Date",
      "Target Date",
      "X2 Predicted",
      "Actual Close",
      "Range Lower",
      "Range Upper",
      "Absolute Error",
      "Status",
    ];

    const rows = history.map((row) => [
      row.base_date || "",
      row.target_date || "",
      row.experimental_x2_point_price ??
        row.predicted_price ??
        "",
      row.actual_close ?? "",
      row.expected_range_lower ?? "",
      row.expected_range_upper ?? "",
      row.experimental_x2_point_absolute_error ??
        row.absolute_error ??
        "",
      row.status || "",
    ]);

    const csv = [
      headers,
      ...rows,
    ]
      .map((row) =>
        row
          .map((value) => {
            const text = String(value);
            return `"${text.replace(
              /"/g,
              '""'
            )}"`;
          })
          .join(",")
      )
      .join("\n");

    const blob = new Blob(
      [csv],
      {
        type: "text/csv;charset=utf-8;",
      }
    );

    const url =
      URL.createObjectURL(blob);

    const anchor =
      document.createElement("a");

    anchor.href = url;
    anchor.download =
      `${symbol.replace(
        ".NS",
        ""
      )}_prediction_history.csv`;

    anchor.click();

    URL.revokeObjectURL(url);
  };


  const resolvedCount =
    history.filter(
      (row) =>
        String(row.status).toUpperCase() ===
        "RESOLVED"
    ).length;


  const pendingCount =
    history.length -
    resolvedCount;


  return (
    <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-5">

      {/* HEADER */}
      <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-center">

        <div>
          <div className="flex items-center gap-2">
            <LineChartIcon
              size={17}
              className="text-violet-400"
            />

            <h2 className="text-sm font-semibold text-white">
              Prediction History Chart
            </h2>
          </div>

          <p className="mt-1 text-[10px] text-gray-600">
            X2 prediction vs actual close with the 80% expected range.
          </p>
        </div>


        <div className="flex flex-wrap items-center gap-2">

          <div className="flex items-center gap-2 rounded-lg border border-white/5 bg-[#0b1018] px-3 py-2">
            <CalendarDays
              size={13}
              className="text-gray-500"
            />

            <select
              value={recordLimit}
              onChange={(event) =>
                setRecordLimit(
                  Number(event.target.value)
                )
              }
              className="bg-transparent text-[10px] text-gray-300 outline-none"
            >
              <option value={5}>
                Last 5 Records
              </option>

              <option value={10}>
                Last 10 Records
              </option>

              <option value={20}>
                Last 20 Records
              </option>

              <option value={50}>
                Last 50 Records
              </option>
            </select>
          </div>


          <button
            onClick={exportCsv}
            className="flex items-center gap-2 rounded-lg border border-white/10 bg-[#0b1018] px-3 py-2 text-[10px] font-medium text-gray-300 transition hover:border-violet-500/30 hover:text-white"
          >
            <Download size={13} />
            Export CSV
          </button>

        </div>
      </div>


      {/* SMALL STATUS BAR */}
      <div className="mt-4 flex flex-wrap gap-2">

        <span className="rounded-full border border-white/5 bg-[#0b1018] px-3 py-1 text-[9px] text-gray-500">
          {history.length} total
        </span>

        <span className="rounded-full border border-green-500/10 bg-green-500/[0.05] px-3 py-1 text-[9px] text-green-400">
          {resolvedCount} resolved
        </span>

        <span className="rounded-full border border-amber-500/10 bg-amber-500/[0.05] px-3 py-1 text-[9px] text-amber-400">
          {pendingCount} pending
        </span>

      </div>


      {/* LEGEND */}
      <div className="mt-5 flex flex-wrap justify-center gap-x-5 gap-y-2 text-[9px]">

        <span className="flex items-center gap-2 text-gray-400">
          <span className="h-2 w-2 rounded-full bg-blue-500" />
          X2 Predicted
        </span>

        <span className="flex items-center gap-2 text-gray-400">
          <span className="h-2 w-2 rounded-full bg-green-500" />
          Actual Close
        </span>

        <span className="flex items-center gap-2 text-gray-400">
          <span className="h-2 w-4 rounded-sm bg-blue-500/20" />
          80% Expected Range
        </span>

      </div>


      {/* CHART */}
      <div className="mt-2 h-[360px] w-full">

        {chartData.length ? (

          <ResponsiveContainer
            width="100%"
            height="100%"
          >

            <ComposedChart
              data={chartData}
              margin={{
                top: 20,
                right: 20,
                left: 0,
                bottom: 0,
              }}
            >

              <CartesianGrid
                strokeDasharray="3 3"
                stroke="#1f2937"
                vertical={true}
              />


              <XAxis
                dataKey="date"
                axisLine={false}
                tickLine={false}
                minTickGap={18}
                tick={{
                  fill: "#64748b",
                  fontSize: 10,
                }}
              />


              <YAxis
                orientation="right"
                axisLine={false}
                tickLine={false}
                width={66}
                domain={[
                  "auto",
                  "auto",
                ]}
                tick={{
                  fill: "#64748b",
                  fontSize: 10,
                }}
                tickFormatter={(value) =>
                  `₹${Number(
                    value
                  ).toLocaleString(
                    "en-IN",
                    {
                      maximumFractionDigits: 0,
                    }
                  )}`
                }
              />


              <Tooltip
                content={
                  <PredictionTooltip />
                }
              />


              {/* Invisible base required for a band between lower and upper */}
              <Area
                type="monotone"
                dataKey="rangeLow"
                stackId="range"
                stroke="none"
                fill="transparent"
                connectNulls
                isAnimationActive={false}
              />


              {/* Shaded 80% range */}
              <Area
                type="monotone"
                dataKey="rangeBand"
                stackId="range"
                name="80% Expected Range"
                stroke="none"
                fill="#3b82f6"
                fillOpacity={0.10}
                connectNulls
                isAnimationActive={false}
              />


              {/* Range lower dashed boundary */}
              <Line
                type="monotone"
                dataKey="rangeLow"
                name="80% Range Lower"
                stroke="#64748b"
                strokeWidth={1.4}
                strokeDasharray="5 5"
                dot={false}
                connectNulls
                isAnimationActive={false}
              />


              {/* Range upper dashed boundary */}
              <Line
                type="monotone"
                dataKey="rangeHigh"
                name="80% Range Upper"
                stroke="#94a3b8"
                strokeWidth={1.4}
                strokeDasharray="5 5"
                dot={false}
                connectNulls
                isAnimationActive={false}
              />


              {/* Experimental X2 */}
              <Line
                type="monotone"
                dataKey="x2Point"
                name="X2 Predicted"
                stroke="#3b82f6"
                strokeWidth={2.5}
                dot={{
                  r: 3,
                  fill: "#3b82f6",
                  stroke: "#dbeafe",
                  strokeWidth: 1,
                }}
                activeDot={{
                  r: 5,
                }}
                connectNulls
                isAnimationActive={false}
              />


              {/* Actual close */}
              <Line
                type="monotone"
                dataKey="actualClose"
                name="Actual Close"
                stroke="#22c55e"
                strokeWidth={2.5}
                dot={{
                  r: 3,
                  fill: "#22c55e",
                  stroke: "#dcfce7",
                  strokeWidth: 1,
                }}
                activeDot={{
                  r: 5,
                }}
                connectNulls
                isAnimationActive={false}
              />

            </ComposedChart>

          </ResponsiveContainer>

        ) : (

          <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-white/5 bg-[#0b1018]/40">
            <div className="text-center">
              <LineChartIcon
                size={28}
                className="mx-auto text-gray-700"
              />

              <p className="mt-3 text-sm font-medium text-gray-400">
                No prediction history yet
              </p>

              <p className="mt-1 text-[10px] text-gray-600">
                Prediction records will appear here after validation history is created.
              </p>
            </div>
          </div>

        )}

      </div>


      {/* FOOTNOTE */}
      <div className="mt-3 rounded-lg border border-blue-500/10 bg-blue-500/[0.03] px-3 py-2.5">
        <p className="text-[9px] leading-4 text-gray-600">
          Blue = experimental X2 point estimate. Green = actual next-trading-day close.
          The shaded/dashed area is the historical 80% expected range.
          Pending predictions do not have an actual-close value yet.
        </p>
      </div>

    </div>
  );
}
