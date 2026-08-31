import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  Activity,
  Bell,
  Bot,
  BarChart3,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Circle,
  Download,
  CalendarDays,
  Gauge,
  Flame,
  Info,
  LayoutDashboard,
  LineChart as LineChartIcon,
  LoaderCircle,
  Maximize2,
  Minimize2,
  Menu,
  Moon,
  Newspaper,
  GitCompareArrows,
  Plus,
  RefreshCw,
  Search,
  Settings,
  Star,
  Sun,
  TrendingDown,
  TrendingUp,
  Wallet,
  XCircle,
} from "lucide-react";

import {
  ResponsiveContainer,
  ComposedChart,
  LineChart,
  Area,
  Bar,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  ReferenceDot,
} from "recharts";

import CandlestickStockChart from "./CandlestickStockChart";


const API_URL =
  "http://127.0.0.1:8000";


// =========================================================
// STOCK LIST
// =========================================================

const STOCKS = [
  {
    name: "Reliance Industries",
    symbol: "RELIANCE.NS",
    short: "RELIANCE",
  },
  {
    name: "Tata Consultancy Services",
    symbol: "TCS.NS",
    short: "TCS",
  },
  {
    name: "Infosys",
    symbol: "INFY.NS",
    short: "INFY",
  },
  {
    name: "HDFC Bank",
    symbol: "HDFCBANK.NS",
    short: "HDFCBANK",
  },
  {
    name: "ICICI Bank",
    symbol: "ICICIBANK.NS",
    short: "ICICIBANK",
  },
  {
    name: "Axis Bank",
    symbol: "AXISBANK.NS",
    short: "AXISBANK",
  },
  {
    name: "Kotak Mahindra Bank",
    symbol: "KOTAKBANK.NS",
    short: "KOTAKBANK",
  },
  {
    name: "Wipro",
    symbol: "WIPRO.NS",
    short: "WIPRO",
  },
  {
    name: "HCL Technologies",
    symbol: "HCLTECH.NS",
    short: "HCLTECH",
  },
  {
    name: "Tata Motors",
    symbol: "TATAMOTORS.NS",
    short: "TATAMOTORS",
  },
  {
    name: "State Bank of India",
    symbol: "SBIN.NS",
    short: "SBIN",
  },
  {
    name: "ITC",
    symbol: "ITC.NS",
    short: "ITC",
  },
  {
    name: "Larsen & Toubro",
    symbol: "LT.NS",
    short: "LT",
  },
  {
    name: "Bharti Airtel",
    symbol: "BHARTIARTL.NS",
    short: "BHARTIARTL",
  },
  {
    name: "Maruti Suzuki",
    symbol: "MARUTI.NS",
    short: "MARUTI",
  },
  {
    name: "Sun Pharmaceutical",
    symbol: "SUNPHARMA.NS",
    short: "SUNPHARMA",
  },
];


const DEFAULT_WATCHLIST_SYMBOLS =
  STOCKS.slice(
    0,
    6
  ).map(
    (
      item
    ) =>
      item.symbol
  );


const RANGE_OPTIONS = [
  {
    key: "1d",
    label: "1D",
  },
  {
    key: "5d",
    label: "1W",
  },
  {
    key: "1mo",
    label: "1M",
  },
  {
    key: "3mo",
    label: "3M",
  },
  {
    key: "6mo",
    label: "6M",
  },
  {
    key: "1y",
    label: "1Y",
  },
  {
    key: "5y",
    label: "5Y",
  },
];


// =========================================================
// CANDLESTICK RANGE CONFIG
// =========================================================

const CANDLE_RANGE_CONFIG = {
  "1d": {
    period: "1d",
    interval: "5m",
  },

  "5d": {
    period: "5d",
    interval: "30m",
  },

  "1mo": {
    period: "1mo",
    interval: "1d",
  },

  "3mo": {
    period: "3mo",
    interval: "1d",
  },

  "6mo": {
    period: "6mo",
    interval: "1d",
  },

  "1y": {
    period: "1y",
    interval: "1d",
  },

  "5y": {
    period: "5y",
    interval: "1wk",
  },
};


// =========================================================
// FORMATTERS
// =========================================================

function formatPrice(
  value
) {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(
      Number(value)
    )
  ) {
    return "--";
  }

  return `₹${Number(
    value
  ).toLocaleString(
    "en-IN",
    {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }
  )}`;
}


function formatNumber(
  value
) {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(
      Number(value)
    )
  ) {
    return "--";
  }

  return Number(
    value
  ).toLocaleString(
    "en-IN",
    {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }
  );
}


function formatPercent(
  value,
  digits = 2
) {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(
      Number(value)
    )
  ) {
    return "--";
  }

  const number =
    Number(value);

  return `${
    number > 0
      ? "+"
      : ""
  }${number.toFixed(
    digits
  )}%`;
}


function formatProbabilityScore(
  value,
  digits = 2
) {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(
      Number(value)
    )
  ) {
    return "--";
  }

  return `${(
    Number(value) *
    100
  ).toFixed(
    digits
  )}%`;
}


function formatVolume(
  value
) {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(
      Number(value)
    )
  ) {
    return "--";
  }

  const number =
    Number(value);

  if (
    number >=
    10000000
  ) {
    return `${(
      number /
      10000000
    ).toFixed(
      2
    )} Cr`;
  }

  if (
    number >=
    100000
  ) {
    return `${(
      number /
      100000
    ).toFixed(
      2
    )} L`;
  }

  if (
    number >=
    1000
  ) {
    return `${(
      number /
      1000
    ).toFixed(
      2
    )} K`;
  }

  return number.toLocaleString(
    "en-IN"
  );
}


// =========================================================
// SYMBOL NORMALIZATION
// =========================================================

function normalizeStockSymbol(
  symbol
) {
  let cleanSymbol =
    String(
      symbol || ""
    )
      .trim()
      .toUpperCase()
      .replace(
        /\s+/g,
        ""
      );

  if (!cleanSymbol) {
    return "";
  }

  if (
    !cleanSymbol.includes(
      "."
    ) &&
    !cleanSymbol.startsWith(
      "^"
    )
  ) {
    cleanSymbol =
      `${cleanSymbol}.NS`;
  }

  return cleanSymbol;
}


// =========================================================
// NAVIGATION ITEM
// =========================================================

function NavItem({
  icon: Icon,
  label,
  active,
  onClick,
}) {
  return (
    <button
      onClick={
        onClick
      }
      className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-sm transition focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/40 ${
        active
          ? "bg-blue-500/15 text-blue-400"
          : "text-gray-400 hover:bg-white/5 hover:text-white"
      }`}
    >
      <Icon
        size={18}
      />

      <span>
        {label}
      </span>
    </button>
  );
}


// =========================================================
// STAT CARD
// =========================================================

function StatCard({
  title,
  value,
  subtitle,
}) {
  return (
    <div className="rounded-xl border border-[#1b2738] bg-[#0f141d] p-4">

      <p className="text-xs text-gray-500">
        {title}
      </p>

      <p className="mt-2 text-xl font-semibold text-white">
        {value}
      </p>

      {subtitle && (
        <p className="mt-1 text-xs text-gray-600">
          {subtitle}
        </p>
      )}

    </div>
  );
}


// =========================================================
// INDICATOR CARD
// =========================================================

function IndicatorCard({
  title,
  value,
  status,
}) {
  return (
    <div className="rounded-xl border border-[#1b2738] bg-[#0f141d] p-4">

      <p className="text-xs text-gray-500">
        {title}
      </p>

      <div className="mt-3 flex items-end justify-between gap-3">

        <p className="text-xl font-semibold text-white">
          {value}
        </p>

        {status && (
          <span className="rounded-full bg-white/5 px-2 py-1 text-[11px] text-gray-400">
            {status}
          </span>
        )}

      </div>

    </div>
  );
}


// =========================================================
// MARKET OVERVIEW CARD
// =========================================================

function MarketOverviewCard({
  title,
  symbol,
  data,
  loading,
}) {
  const changePercent =
    Number(
      data?.change_percent ||
      0
    );


  const positive =
    changePercent >= 0;


  const miniData =
    Array.isArray(
      data?.chart
    )
      ? data.chart.slice(
          -45
        )
      : [];


  return (
    <div className="relative h-[108px] overflow-hidden rounded-xl border border-[#1b2738] bg-[linear-gradient(145deg,#111826_0%,#0d1420_100%)] px-4 py-3 shadow-[0_10px_30px_rgba(0,0,0,0.16)]">

      <div className="flex items-start justify-between gap-3">

        <div>

          <p className="text-[11px] font-semibold tracking-wide text-gray-300">
            {title}
          </p>

          <p className="mt-1 text-[9px] text-gray-600">
            {symbol}
          </p>

        </div>


        <p
          className={`text-sm font-semibold ${
            positive
              ? "text-green-400"
              : "text-red-400"
          }`}
        >
          {data
            ? formatPercent(
                changePercent
              )
            : "--"}
        </p>

      </div>


      <p className="relative z-10 mt-1 text-[21px] font-semibold leading-none text-white">

        {loading &&
        !data
          ? "Loading..."
          : formatNumber(
              data?.price
            )}

      </p>


      <div className="absolute bottom-2 left-3 right-3 flex items-end gap-3">

        <div className="h-8 min-w-0 flex-1">

          {miniData.length > 2 ? (

            <ResponsiveContainer
              width="100%"
              height="100%"
            >

              <ComposedChart
                data={
                  miniData
                }
                margin={{
                  top: 2,
                  right: 0,
                  left: 0,
                  bottom: 0,
                }}
              >

                <YAxis
                  hide
                  domain={[
                    "dataMin",
                    "dataMax",
                  ]}
                />

                <Line
                  type="monotone"
                  dataKey="price"
                  stroke={
                    positive
                      ? "#4ade80"
                      : "#fb4b55"
                  }
                  strokeWidth={1.8}
                  dot={false}
                  isAnimationActive={false}
                />

              </ComposedChart>

            </ResponsiveContainer>

          ) : (

            <div
              className={`mt-4 h-[2px] w-full rounded-full ${
                positive
                  ? "bg-green-400/70"
                  : "bg-red-400/70"
              }`}
            />

          )}

        </div>


        <span
          className={`mb-0.5 shrink-0 rounded-md px-2 py-1 text-[9px] font-semibold ${
            positive
              ? "bg-green-500/10 text-green-400"
              : "bg-red-500/10 text-red-400"
          }`}
        >
          {positive
            ? "Bullish ↑"
            : "Bearish ↓"}
        </span>

      </div>

    </div>
  );
}

// =========================================================
// MARKET STATUS CARD
// =========================================================

function MarketStatusCard({
  data,
  loading,
}) {
  const values =
    Object.values(
      data || {}
    )
      .map(
        (item) =>
          Number(
            item?.change_percent
          )
      )
      .filter(
        (value) =>
          Number.isFinite(
            value
          )
      );


  const average =
    values.length
      ? values.reduce(
          (sum, value) =>
            sum + value,
          0
        ) / values.length
      : 0;


  const status =
    average > 0.15
      ? "Bullish"
      : average < -0.15
      ? "Bearish"
      : "Mixed";


  const bullish =
    status === "Bullish";


  const bearish =
    status === "Bearish";


  return (
    <div className="h-[108px] rounded-xl border border-[#1b2738] bg-[linear-gradient(145deg,#111826_0%,#0d1420_100%)] px-4 py-3 shadow-[0_10px_30px_rgba(0,0,0,0.16)]">

      <div className="flex h-full items-center justify-between gap-3">

        <div>

          <p className="text-[11px] font-semibold text-gray-300">
            Market Status
          </p>


          <p
            className={`mt-1 text-xl font-bold ${
              bullish
                ? "text-green-400"
                : bearish
                ? "text-red-400"
                : "text-yellow-400"
            }`}
          >
            {loading &&
            values.length === 0
              ? "Loading..."
              : status}
          </p>


          <p className="mt-1 text-[9px] text-gray-500">

            {bullish
              ? "Overall market is bullish"
              : bearish
              ? "Overall market is bearish"
              : "Major indices are mixed"}

          </p>


          <p className="mt-1 text-[8px] text-gray-700">
            Avg. move{" "}
            {values.length
              ? formatPercent(
                  average
                )
              : "--"}
          </p>

        </div>


        <div
          className={`flex h-14 w-14 items-center justify-center rounded-2xl ${
            bullish
              ? "bg-green-500/[0.07]"
              : bearish
              ? "bg-red-500/[0.07]"
              : "bg-yellow-500/[0.07]"
          }`}
        >

          {bullish ? (
            <TrendingUp
              size={32}
              className="text-green-400"
            />
          ) : bearish ? (
            <TrendingDown
              size={32}
              className="text-red-400"
            />
          ) : (
            <Gauge
              size={30}
              className="text-yellow-400"
            />
          )}

        </div>

      </div>

    </div>
  );
}

// =========================================================
// TRACKED MARKET BREADTH
// =========================================================

function MarketBreadthCard({
  items,
  loading,
}) {
  const valid =
    (items || []).filter(
      (item) =>
        Number.isFinite(
          Number(
            item?.change_percent
          )
        )
    );


  const advances =
    valid.filter(
      (item) =>
        Number(
          item.change_percent
        ) >= 0
    ).length;


  const declines =
    Math.max(
      0,
      valid.length -
      advances
    );


  const advanceWidth =
    valid.length
      ? (
          advances /
          valid.length
        ) * 100
      : 50;


  return (
    <div className="h-[108px] rounded-xl border border-[#1b2738] bg-[linear-gradient(145deg,#111826_0%,#0d1420_100%)] px-4 py-3 shadow-[0_10px_30px_rgba(0,0,0,0.16)]">

      <p className="text-[11px] font-semibold text-gray-300">
        Tracked Stocks Breadth
      </p>


      <div className="mt-2 flex items-center justify-between">

        <div>

          <p className="text-xl font-bold text-green-400">
            {loading &&
            valid.length === 0
              ? "..."
              : advances}
          </p>

          <p className="text-[9px] text-green-500/80">
            Up
          </p>

        </div>


        <div className="text-right">

          <p className="text-xl font-bold text-red-400">
            {loading &&
            valid.length === 0
              ? "..."
              : declines}
          </p>

          <p className="text-[9px] text-red-500/80">
            Down
          </p>

        </div>

      </div>


      <div className="mt-2 flex h-1.5 overflow-hidden rounded-full bg-white/5">

        <div
          className="h-full bg-green-400"
          style={{
            width:
              `${advanceWidth}%`,
          }}
        />

        <div className="h-full flex-1 bg-red-400" />

      </div>

    </div>
  );
}

// =========================================================
// LOADING BOX
// =========================================================

function LoadingBox({
  text = "Loading...",
}) {
  return (
    <div className="flex min-h-72 items-center justify-center rounded-2xl border border-white/5 bg-[#0b1018]">

      <div className="text-center">

        <LoaderCircle
          size={34}
          className="mx-auto animate-spin text-blue-400"
        />

        <p className="mt-4 text-sm text-gray-500">
          {text}
        </p>

      </div>

    </div>
  );
}


// =========================================================
// TREND BADGE
// =========================================================

function TrendBadge({
  signal,
}) {
  const normalized =
    String(
      signal ||
      "NEUTRAL"
    ).toUpperCase();


  if (
    normalized ===
    "BULLISH"
  ) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-green-500/10 px-3 py-1 text-xs font-medium text-green-400">

        <TrendingUp
          size={13}
        />

        BULLISH

      </span>
    );
  }


  if (
    normalized ===
    "BEARISH"
  ) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-red-500/10 px-3 py-1 text-xs font-medium text-red-400">

        <TrendingDown
          size={13}
        />

        BEARISH

      </span>
    );
  }


  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-yellow-500/10 px-3 py-1 text-xs font-medium text-yellow-400">

      <Gauge
        size={13}
      />

      NEUTRAL

    </span>
  );
}



// =========================================================
// V9 RELATIVE-STRENGTH BADGE
// =========================================================

function RelativeStrengthBadge({
  signal,
}) {
  const normalized =
    String(
      signal ||
      "NEUTRAL"
    ).toUpperCase();


  if (
    normalized ===
    "OUTPERFORM"
  ) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-green-500/10 px-3 py-1 text-xs font-medium text-green-400">

        <TrendingUp
          size={13}
        />

        OUTPERFORM

      </span>
    );
  }


  if (
    normalized ===
    "UNDERPERFORM"
  ) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-red-500/10 px-3 py-1 text-xs font-medium text-red-400">

        <TrendingDown
          size={13}
        />

        UNDERPERFORM

      </span>
    );
  }


  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-yellow-500/10 px-3 py-1 text-xs font-medium text-yellow-400">

      <Gauge
        size={13}
      />

      NEUTRAL

    </span>
  );
}


// =========================================================
// V9 RELATIVE-STRENGTH CONTENT
// =========================================================

function RelativeStrengthContent({
  data,
  loading,
  error,
  compact = false,
}) {
  if (loading) {
    return (
      <LoadingBox
        text="Running V9 relative-strength analysis..."
      />
    );
  }


  if (error) {
    return (
      <div className="rounded-xl border border-red-500/20 bg-red-500/10 p-5">

        <p className="text-sm font-medium text-red-400">
          Relative-strength analysis unavailable
        </p>

        <p className="mt-2 text-xs leading-5 text-red-300/70">
          {error}
        </p>

      </div>
    );
  }


  if (!data) {
    return (
      <div className="rounded-xl bg-white/5 p-5 text-sm text-gray-500">
        No relative-strength result available.
      </div>
    );
  }


  const probabilities =
    data.probabilities ||
    {};


  const scores = [
    {
      key: "underperform",
      label: "Underperform",
      value:
        Number(
          probabilities.underperform ||
          0
        ),
      bar:
        "bg-red-400",
      text:
        "text-red-400",
    },
    {
      key: "neutral",
      label: "Neutral",
      value:
        Number(
          probabilities.neutral ||
          0
        ),
      bar:
        "bg-yellow-400",
      text:
        "text-yellow-400",
    },
    {
      key: "outperform",
      label: "Outperform",
      value:
        Number(
          probabilities.outperform ||
          0
        ),
      bar:
        "bg-green-400",
      text:
        "text-green-400",
    },
  ];


  const sortedScores = [
    ...scores,
  ].sort(
    (a, b) =>
      b.value -
      a.value
  );


  const topScore =
    sortedScores[0]
      ?.value ||
    0;


  const secondScore =
    sortedScores[1]
      ?.value ||
    0;


  const probabilityGap =
    data.top_probability_gap !==
      undefined
      ? Number(
          data.top_probability_gap
        )
      : topScore -
        secondScore;


  const signalStrength =
    probabilityGap >=
    0.12
      ? "Clearer signal"
      : probabilityGap >=
        0.06
      ? "Moderate signal"
      : "Close signal";


  const context =
    data.current_market_context ||
    {};


  return (
    <div>

      <div className="flex flex-wrap items-start justify-between gap-4">

        <div>

          <p className="text-xs text-gray-500">
            5-Day Relative Outlook vs NIFTY 50
          </p>

          <div className="mt-3 flex flex-wrap items-center gap-3">

            <RelativeStrengthBadge
              signal={
                data.signal
              }
            />

            <span className="rounded-full bg-white/5 px-2.5 py-1 text-[11px] text-gray-400">
              {signalStrength}
            </span>

          </div>

        </div>


        <div className="text-right">

          <p className="text-xs text-gray-600">
            Top raw score
          </p>

          <p className="mt-1 text-2xl font-bold text-white">
            {formatProbabilityScore(
              data.top_probability
            )}
          </p>

          <p className="mt-1 text-[10px] text-gray-600">
            Not calibrated confidence
          </p>

        </div>

      </div>


      <div className="mt-6 space-y-4">

        {scores.map(
          (
            score
          ) => (

            <div
              key={
                score.key
              }
            >

              <div className="mb-1.5 flex items-center justify-between">

                <span className="text-xs text-gray-500">
                  {score.label}
                </span>

                <span
                  className={`text-xs font-semibold ${score.text}`}
                >
                  {formatProbabilityScore(
                    score.value
                  )}
                </span>

              </div>


              <div className="h-1.5 overflow-hidden rounded-full bg-white/5">

                <div
                  className={`h-full rounded-full ${score.bar}`}
                  style={{
                    width:
                      `${Math.max(
                        0,
                        Math.min(
                          100,
                          score.value *
                            100
                        )
                      )}%`,
                  }}
                />

              </div>

            </div>

          )
        )}

      </div>


      {!compact && (

        <div className="mt-6 grid gap-3 sm:grid-cols-3">

          <StatCard
            title="Stock Previous 5D"
            value={
              formatPercent(
                context.stock_previous_5d_return_percent
              )
            }
          />

          <StatCard
            title="NIFTY Previous 5D"
            value={
              formatPercent(
                context.nifty_previous_5d_return_percent
              )
            }
          />

          <StatCard
            title="Previous 5D Relative Strength"
            value={
              formatPercent(
                context.previous_5d_relative_strength_percent
              )
            }
          />

        </div>

      )}


      <div className="mt-5 rounded-xl border border-blue-500/10 bg-blue-500/5 p-4">

        <p className="text-xs leading-5 text-gray-500">

          V9 predicts whether the stock may outperform, move roughly
          in line with, or underperform NIFTY 50 over the next 5
          trading days. Probability values are raw model scores and
          should not be interpreted as certainty.

        </p>

      </div>


      {!data.research_scope
        ?.stock_in_original_15_stock_evaluation && (

        <div className="mt-3 rounded-xl border border-yellow-500/10 bg-yellow-500/5 p-4">

          <p className="text-xs leading-5 text-yellow-200/70">

            This ticker was not part of the original 15-stock V9
            walk-forward evaluation universe, so the historical V9
            performance metrics should not be assumed to apply equally
            to this symbol.

          </p>

        </div>

      )}

    </div>
  );
}


// =========================================================
// MARKET TOOLTIP
// =========================================================

function MarketChartTooltip({
  active,
  payload,
  label,
}) {
  if (
    !active ||
    !payload ||
    payload.length === 0
  ) {
    return null;
  }

  const point =
    payload[0]?.payload;

  if (!point) {
    return null;
  }

  return (
    <div className="rounded-xl border border-white/10 bg-[#090d14] px-4 py-3 shadow-2xl">

      <p className="text-xs font-medium text-gray-400">
        {label}
      </p>

      <p className="mt-2 text-sm font-semibold text-white">
        Price:{" "}
        {formatPrice(
          point.price
        )}
      </p>

      <p className="mt-1 text-xs text-gray-500">
        Volume:{" "}
        {formatVolume(
          point.volume
        )}
      </p>

    </div>
  );
}


// =========================================================
// MARKET PRICE CHART
// =========================================================

function YahooStyleChart({
  stock,
  selectedRange,
}) {
  if (
    !stock?.chart ||
    stock.chart.length === 0
  ) {
    return (
      <div className="flex h-80 items-center justify-center text-sm text-gray-500">
        No chart data available
      </div>
    );
  }


  const previousClose =
    Number(
      stock.previous_close ||
      0
    );


  const lastPoint =
    stock.chart[
      stock.chart.length -
      1
    ];


  const currentPrice =
    Number(
      lastPoint?.price ||
      stock.price ||
      0
    );


  const prices =
    stock.chart.map(
      (item) =>
        Number(
          item.price
        )
    );


  const volumes =
    stock.chart.map(
      (item) =>
        Number(
          item.volume ||
          0
        )
    );


  const rawMin =
    Math.min(
      ...prices,
      previousClose
    );


  const rawMax =
    Math.max(
      ...prices,
      previousClose
    );


  const priceRange =
    rawMax -
      rawMin ||
    1;


  const padding =
    priceRange *
    0.12;


  const yMin =
    rawMin -
    padding;


  const yMax =
    rawMax +
    padding;


  let previousCloseOffset =
    (
      (
        yMax -
        previousClose
      ) /
      (
        yMax -
        yMin
      )
    ) *
    100;


  previousCloseOffset =
    Math.max(
      0,
      Math.min(
        100,
        previousCloseOffset
      )
    );


  const maxVolume =
    Math.max(
      ...volumes,
      1
    );


  const currentColor =
    currentPrice >=
    previousClose
      ? "#10b981"
      : "#ef4444";


  return (
    <div className="h-[292px] w-full">

      <ResponsiveContainer
        width="100%"
        height="100%"
      >

        <ComposedChart
          data={
            stock.chart
          }
          margin={{
            top: 20,
            right: 68,
            left: 5,
            bottom: 5,
          }}
        >

          <defs>

            <linearGradient
              id="dynamicPriceStroke"
              x1="0"
              y1="0"
              x2="0"
              y2="1"
            >

              <stop
                offset="0%"
                stopColor="#10b981"
              />

              <stop
                offset={`${previousCloseOffset}%`}
                stopColor="#10b981"
              />

              <stop
                offset={`${previousCloseOffset}%`}
                stopColor="#ef4444"
              />

              <stop
                offset="100%"
                stopColor="#ef4444"
              />

            </linearGradient>


            <linearGradient
              id="dynamicPriceFill"
              x1="0"
              y1="0"
              x2="0"
              y2="1"
            >

              <stop
                offset="0%"
                stopColor="#10b981"
                stopOpacity={0.18}
              />

              <stop
                offset={`${previousCloseOffset}%`}
                stopColor="#10b981"
                stopOpacity={0.04}
              />

              <stop
                offset={`${previousCloseOffset}%`}
                stopColor="#ef4444"
                stopOpacity={0.04}
              />

              <stop
                offset="100%"
                stopColor="#ef4444"
                stopOpacity={0.15}
              />

            </linearGradient>

          </defs>


          <CartesianGrid
            stroke="#1d2531"
            strokeDasharray="3 3"
            vertical={false}
          />


          <XAxis
            dataKey="time"
            axisLine={false}
            tickLine={false}
            minTickGap={45}
            tick={{
              fill: "#6b7280",
              fontSize: 11,
            }}
          />


          <YAxis
            yAxisId="price"
            orientation="right"
            domain={[
              yMin,
              yMax,
            ]}
            axisLine={false}
            tickLine={false}
            width={60}
            tick={{
              fill: "#6b7280",
              fontSize: 11,
            }}
            tickFormatter={
              (value) =>
                Number(
                  value
                ).toFixed(
                  0
                )
            }
          />


          <YAxis
            yAxisId="volume"
            hide
            domain={[
              0,
              maxVolume *
                6,
            ]}
          />


          <Tooltip
            content={
              <MarketChartTooltip />
            }
          />


          <Bar
            yAxisId="volume"
            dataKey="volume"
            fill="#6b7280"
            opacity={0.35}
            barSize={
              selectedRange ===
              "1d"
                ? 4
                : 7
            }
            isAnimationActive={
              false
            }
          />


          <ReferenceLine
            yAxisId="price"
            y={
              previousClose
            }
            stroke="#d1d5db"
            strokeDasharray="6 6"
            strokeWidth={1}
            opacity={0.75}

          />


          <Area
            yAxisId="price"
            type="linear"
            dataKey="price"
            stroke="url(#dynamicPriceStroke)"
            strokeWidth={2}
            fill="url(#dynamicPriceFill)"
            dot={false}
            isAnimationActive={
              false
            }
          />


          <ReferenceLine
            yAxisId="price"
            y={
              currentPrice
            }
            stroke={
              currentColor
            }
            strokeDasharray="4 5"
            opacity={0.5}

          />


          <ReferenceDot
            yAxisId="price"
            x={
              lastPoint.time
            }
            y={
              currentPrice
            }
            r={4}
            fill={
              currentColor
            }
            stroke={
              currentColor
            }
            isFront
          />

        </ComposedChart>

      </ResponsiveContainer>

    </div>
  );
}


// =========================================================
// FUTURE FORECAST TOOLTIP
// =========================================================

function ForecastTooltip({
  active,
  payload,
}) {
  if (
    !active ||
    !payload ||
    payload.length === 0
  ) {
    return null;
  }


  const point =
    payload[0]?.payload;


  if (!point) {
    return null;
  }


  return (
    <div className="min-w-60 rounded-xl border border-white/10 bg-[#080d15] p-4 shadow-2xl">

      <p className="text-xs font-semibold text-blue-400">

        {point.horizon ===
        "NOW"
          ? "Current Position"
          : `${point.horizon} Forecast`}

      </p>


      <div className="mt-4 space-y-3">

        <div className="flex items-center justify-between gap-6">

          <span className="text-xs text-gray-500">
            Expected Move
          </span>

          <span
            className={`text-sm font-semibold ${
              Number(
                point.expected
              ) >= 0
                ? "text-green-400"
                : "text-red-400"
            }`}
          >
            {formatPercent(
              point.expected
            )}
          </span>

        </div>


        <div className="flex items-center justify-between gap-6">

          <span className="text-xs text-gray-500">
            Predicted Price
          </span>

          <span className="text-sm font-medium text-white">
            {formatPrice(
              point.predictedPrice
            )}
          </span>

        </div>


        {point.horizon !==
          "NOW" && (
          <>

            <div className="border-t border-white/5 pt-3">

              <p className="text-[11px] text-gray-500">
                80% Estimated Range
              </p>

              <p className="mt-1 text-xs text-gray-300">

                {formatPercent(
                  point.lower
                )}

                {" → "}

                {formatPercent(
                  point.upper
                )}

              </p>

            </div>


            <div className="flex items-center justify-between">

              <span className="text-xs text-gray-500">
                Signal
              </span>

              <span className="text-xs font-semibold text-gray-300">
                {point.signal}
              </span>

            </div>

          </>
        )}

      </div>

    </div>
  );
}


// =========================================================
// IMPROVED FUTURE FORECAST CHART
// =========================================================

function FutureForecastChart({
  forecast,
}) {
  if (
    !forecast?.forecasts ||
    forecast.forecasts.length ===
      0
  ) {
    return (
      <div className="flex h-[420px] items-center justify-center text-sm text-gray-500">
        No forecast available
      </div>
    );
  }


  // =======================================================
  // BUILD CHART DATA
  // =======================================================

  const chartData = [
    {
      horizon: "NOW",

      expected: 0,

      lower: 0,

      upper: 0,

      predictedPrice:
        forecast.current_close,

      signal: "CURRENT",
    },


    ...forecast.forecasts.map(
      (item) => ({

        horizon:
          item.horizon,

        expected:
          Number(
            item.expected_move_percent
          ),

        lower:
          Number(
            item
              .estimated_range_80
              ?.lower_percent
          ),

        upper:
          Number(
            item
              .estimated_range_80
              ?.upper_percent
          ),

        predictedPrice:
          Number(
            item.predicted_price
          ),

        signal:
          item.signal,

      })
    ),
  ];


  // =======================================================
  // SYMMETRICAL Y AXIS
  // =======================================================

  const allAbsoluteValues =
    chartData.flatMap(
      (item) => [

        Math.abs(
          Number(
            item.expected ||
            0
          )
        ),

        Math.abs(
          Number(
            item.lower ||
            0
          )
        ),

        Math.abs(
          Number(
            item.upper ||
            0
          )
        ),

      ]
    );


  const maxAbsolute =
    Math.max(
      ...allAbsoluteValues,
      1
    );


  const axisLimit =
    Math.ceil(
      maxAbsolute *
      1.15 *
      10
    ) /
    10;


  // =======================================================
  // RANGE ARRAY
  // =======================================================

  const graphData =
    chartData.map(
      (item) => ({

        ...item,

        uncertaintyRange: [
          item.lower,
          item.upper,
        ],

      })
    );


  // =======================================================
  // CUSTOM DOT
  // =======================================================

  function ForecastDot(
    props
  ) {
    const {
      cx,
      cy,
      payload,
    } = props;


    if (
      cx === undefined ||
      cy === undefined
    ) {
      return null;
    }


    const value =
      Number(
        payload.expected
      );


    const dotColor =
      value > 0
        ? "#10b981"
        : value < 0
        ? "#ef4444"
        : "#94a3b8";


    return (
      <g>

        <circle
          cx={cx}
          cy={cy}
          r={6}
          fill="#0f141d"
          stroke={
            dotColor
          }
          strokeWidth={3}
        />


        <text
          x={cx}
          y={cy - 17}
          textAnchor="middle"
          fill={
            dotColor
          }
          fontSize="11"
          fontWeight="600"
        >

          {payload.horizon ===
          "NOW"
            ? "0.00%"
            : formatPercent(
                value,
                2
              )}

        </text>

      </g>
    );
  }


  return (
    <div className="h-[480px] w-full">

      <ResponsiveContainer
        width="100%"
        height="100%"
      >

        <ComposedChart
          data={
            graphData
          }
          margin={{
            top: 50,
            right: 35,
            left: 10,
            bottom: 20,
          }}
        >

          {/* ===============================================
              GRADIENTS
          =============================================== */}

          <defs>

            <linearGradient
              id="futureRangeGradient"
              x1="0"
              y1="0"
              x2="0"
              y2="1"
            >

              <stop
                offset="0%"
                stopColor="#60a5fa"
                stopOpacity={0.14}
              />

              <stop
                offset="50%"
                stopColor="#60a5fa"
                stopOpacity={0.07}
              />

              <stop
                offset="100%"
                stopColor="#60a5fa"
                stopOpacity={0.025}
              />

            </linearGradient>


            <linearGradient
              id="forecastLineGradient"
              x1="0"
              y1="0"
              x2="1"
              y2="0"
            >

              <stop
                offset="0%"
                stopColor="#94a3b8"
              />

              <stop
                offset="40%"
                stopColor="#ef4444"
              />

              <stop
                offset="60%"
                stopColor="#ef4444"
              />

              <stop
                offset="100%"
                stopColor="#10b981"
              />

            </linearGradient>

          </defs>


          {/* ===============================================
              GRID
          =============================================== */}

          <CartesianGrid
            stroke="#1d2938"
            strokeDasharray="4 4"
            vertical={false}
          />


          {/* ===============================================
              X AXIS
          =============================================== */}

          <XAxis
            dataKey="horizon"
            axisLine={false}
            tickLine={false}
            padding={{
              left: 25,
              right: 25,
            }}
            tick={{
              fill: "#94a3b8",
              fontSize: 12,
              fontWeight: 500,
            }}
          />


          {/* ===============================================
              Y AXIS
          =============================================== */}

          <YAxis
            domain={[
              -axisLimit,
              axisLimit,
            ]}
            axisLine={false}
            tickLine={false}
            width={70}
            tickCount={5}
            allowDecimals
            tick={{
              fill: "#64748b",
              fontSize: 11,
            }}
            tickFormatter={
              (value) =>
                `${
                  Number(
                    value
                  ) > 0
                    ? "+"
                    : ""
                }${Number(
                  value
                ).toFixed(
                  1
                )}%`
            }
          />


          {/* ===============================================
              TOOLTIP
          =============================================== */}

          <Tooltip
            content={
              <ForecastTooltip />
            }
            cursor={{
              stroke:
                "#334155",
              strokeDasharray:
                "4 4",
            }}
          />


          {/* ===============================================
              ZERO BASELINE
          =============================================== */}

          <ReferenceLine
            y={0}
            stroke="#cbd5e1"
            strokeWidth={1.4}
            strokeDasharray="7 7"
            opacity={0.65}
            label={{
              value:
                "0% Baseline",
              position:
                "insideRight",
              fill:
                "#94a3b8",
              fontSize: 10,
            }}
          />


          {/* ===============================================
              VALIDATION RANGE
          =============================================== */}

          <Area
            type="monotone"
            dataKey="uncertaintyRange"
            stroke="none"
            fill="url(#futureRangeGradient)"
            connectNulls
            isAnimationActive={
              false
            }
          />


          {/* ===============================================
              EXPECTED MOVE
          =============================================== */}

          <Line
            type="monotone"
            dataKey="expected"
            stroke="url(#forecastLineGradient)"
            strokeWidth={3}
            dot={
              <ForecastDot />
            }
            activeDot={{
              r: 8,
              stroke:
                "#ffffff",
              strokeWidth: 2,
            }}
            isAnimationActive={
              false
            }
          />

        </ComposedChart>

      </ResponsiveContainer>

    </div>
  );
}


// =========================================================
// FORECAST CARD
// =========================================================

function ForecastCard({
  item,
}) {
  const move =
    Number(
      item.expected_move_percent
    );


  const range =
    item.estimated_range_80;


  return (
    <div className="rounded-xl border border-[#1b2738] bg-[#0f141d] p-5 transition hover:border-white/10">

      <div className="flex items-start justify-between gap-3">

        <div>

          <p className="text-xs font-medium text-gray-500">
            {item.horizon} FORECAST
          </p>


          <p
            className={`mt-2 text-2xl font-bold ${
              move >= 0
                ? "text-green-400"
                : "text-red-400"
            }`}
          >
            {formatPercent(
              move
            )}
          </p>

        </div>


        <TrendBadge
          signal={
            item.signal
          }
        />

      </div>


      <div className="mt-5 space-y-3 border-t border-white/5 pt-4">

        <div className="flex items-center justify-between">

          <span className="text-xs text-gray-500">
            Predicted Price
          </span>

          <span className="text-sm font-medium text-white">
            {formatPrice(
              item.predicted_price
            )}
          </span>

        </div>


        <div>

          <p className="text-xs text-gray-500">
            80% Estimated Range
          </p>

          <p className="mt-1 text-sm text-gray-300">

            {formatPercent(
              range?.lower_percent
            )}

            {" → "}

            {formatPercent(
              range?.upper_percent
            )}

          </p>

        </div>


        <div className="flex items-center justify-between">

          <span className="text-xs text-gray-500">
            Direction Accuracy
          </span>

          <span className="text-sm text-gray-300">

            {item.test_direction_accuracy_percent !==
            null
              ? `${Number(
                  item.test_direction_accuracy_percent
                ).toFixed(
                  2
                )}%`
              : "--"}

          </span>

        </div>


        <div className="flex items-center justify-between">

          <span className="text-xs text-gray-500">
            Model Status
          </span>

          <span
            className={`text-xs font-semibold ${
              item.evaluation_status ===
              "STRONGER"
                ? "text-green-400"
                : item.evaluation_status ===
                  "MODERATE"
                ? "text-yellow-400"
                : "text-orange-400"
            }`}
          >
            {item.evaluation_status ||
              "--"}
          </span>

        </div>


        <div className="flex items-center justify-between">

          <span className="text-xs text-gray-500">
            Baseline
          </span>

          <span
            className={`text-xs font-medium ${
              item.beats_naive_baseline
                ? "text-green-400"
                : "text-orange-400"
            }`}
          >
            {item.beats_naive_baseline
              ? "Beats baseline"
              : "Below baseline"}
          </span>

        </div>

      </div>

    </div>
  );
}


// =========================================================
// TRAINING PROGRESS
// =========================================================

function TrainingProgress({
  status,
  symbol,
}) {
  const stage =
    status?.stage ||
    "Starting";


  const statusValue =
    status?.status ||
    "training";


  const stageOrder = {
    Starting: 0,

    "Validating stock": 1,

    "Preparing data": 2,

    "Training BiLSTM": 3,

    "Evaluating model": 4,

    Ready: 5,
  };


  const currentStage =
    stageOrder[
      stage
    ] ?? 0;


  const steps = [
    {
      label:
        "Validate market data",
      level: 1,
    },
    {
      label:
        "Prepare technical features",
      level: 2,
    },
    {
      label:
        "Train Multi-Horizon BiLSTM",
      level: 3,
    },
    {
      label:
        "Evaluate model performance",
      level: 4,
    },
    {
      label:
        "Forecast model ready",
      level: 5,
    },
  ];


  return (
    <div className="rounded-3xl border border-blue-500/10 bg-[#0f141d] p-7 lg:p-10">

      <div className="mx-auto max-w-2xl">


        {/* HEADER */}

        <div className="text-center">

          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-500/10">

            {statusValue ===
            "error" ? (

              <XCircle
                size={30}
                className="text-red-400"
              />

            ) : (

              <LoaderCircle
                size={30}
                className="animate-spin text-blue-400"
              />

            )}

          </div>


          <h2 className="mt-5 text-2xl font-bold text-white">

            {statusValue ===
            "error"
              ? "Training Failed"
              : "Preparing AI Forecast"}

          </h2>


          <p className="mt-2 text-sm font-medium text-blue-400">
            {symbol}
          </p>


          <p className="mt-3 text-sm leading-6 text-gray-500">

            {status?.message ||
              "Preparing the Multi-Horizon BiLSTM model."}

          </p>

        </div>


        {/* STEPS */}

        <div className="mt-8 space-y-3">

          {steps.map(
            (
              step
            ) => {

              const completed =
                currentStage >
                step.level;


              const active =
                currentStage ===
                step.level;


              const ready =
                currentStage >=
                  5 &&
                step.level ===
                  5;


              return (
                <div
                  key={
                    step.label
                  }
                  className={`flex items-center gap-4 rounded-xl border px-4 py-4 ${
                    active
                      ? "border-blue-500/20 bg-blue-500/5"
                      : completed ||
                        ready
                      ? "border-green-500/10 bg-green-500/[0.03]"
                      : "border-white/5 bg-[#0b1018]"
                  }`}
                >

                  {completed ||
                  ready ? (

                    <CheckCircle2
                      size={20}
                      className="shrink-0 text-green-400"
                    />

                  ) : active ? (

                    <LoaderCircle
                      size={20}
                      className="shrink-0 animate-spin text-blue-400"
                    />

                  ) : (

                    <Circle
                      size={20}
                      className="shrink-0 text-gray-700"
                    />

                  )}


                  <span
                    className={`text-sm ${
                      completed ||
                      ready
                        ? "text-gray-300"
                        : active
                        ? "font-medium text-white"
                        : "text-gray-600"
                    }`}
                  >
                    {step.label}
                  </span>

                </div>
              );
            }
          )}

        </div>


        {statusValue !==
          "error" && (

          <div className="mt-7 rounded-xl border border-yellow-500/10 bg-yellow-500/5 p-4">

            <p className="text-xs leading-5 text-gray-500">

              First-time forecasting for a new ticker requires model
              training. Once the model is saved, future forecasts load
              much faster.

            </p>

          </div>

        )}

      </div>

    </div>
  );
}


// =========================================================
// NEXT-DAY PREDICTION CONTENT
// =========================================================

function PredictionContent({
  prediction,
  loading,
  error,
}) {
  if (loading) {
    return (
      <LoadingBox
        text="Running X2 forecast..."
      />
    );
  }


  if (error) {
    return (
      <div className="rounded-xl border border-red-500/20 bg-red-500/10 p-5">

        <p className="text-sm font-medium text-red-400">
          Prediction unavailable
        </p>

        <p className="mt-2 text-xs leading-5 text-red-300/70">
          {error}
        </p>

      </div>
    );
  }


  if (!prediction) {
    return (
      <div className="rounded-xl bg-white/5 p-5 text-sm text-gray-500">
        No prediction available.
      </div>
    );
  }


  const x2Point =
    prediction?.experimental_x2_point ||
    {};


  const x2Range =
    prediction?.expected_range ||
    {};


  const pointPrice =
    Number(
      x2Point?.price ??
      prediction?.predicted_price
    );


  const move =
    Number(
      x2Point?.move_percent ??
      prediction?.predicted_return_percent ??
      0
    );


  const signal =
    prediction.trend_signal ||
    (
      move >= 0.10
        ? "BULLISH"
        : move <= -0.10
        ? "BEARISH"
        : "NEUTRAL"
    );


  const lower =
    Number(
      x2Range?.lower
    );


  const upper =
    Number(
      x2Range?.upper
    );


  return (
    <div>

      <div className="flex flex-wrap items-center justify-between gap-3">

        <div>

          <p className="text-xs text-gray-500">
            X2 Expected Next-Day Move
          </p>

          <p
            className={`mt-2 text-3xl font-bold ${
              move >= 0
                ? "text-green-400"
                : "text-red-400"
            }`}
          >
            {formatPercent(
              move,
              4
            )}
          </p>

        </div>


        <TrendBadge
          signal={
            signal
          }
        />

      </div>


      <div className="mt-6 grid gap-3 sm:grid-cols-2">

        <StatCard
          title="Latest Daily Close"
          value={
            formatPrice(
              prediction.current_close
            )
          }
        />


        <StatCard
          title="X2 Expected Close"
          value={
            Number.isFinite(
              pointPrice
            )
              ? formatPrice(
                  pointPrice
                )
              : "--"
          }
          subtitle="Experimental point"
        />


        <StatCard
          title="80% Expected Range"
          value={
            Number.isFinite(
              lower
            ) &&
            Number.isFinite(
              upper
            )
              ? `${formatPrice(
                  lower
                )} – ${formatPrice(
                  upper
                )}`
              : "--"
          }
          subtitle="Empirical interval"
        />


        <StatCard
          title="Historical Within ₹20"
          value={
            prediction?.historical_error_profile
              ?.within_20_percent !==
              null &&
            prediction?.historical_error_profile
              ?.within_20_percent !==
              undefined
              ? `${Number(
                  prediction.historical_error_profile
                    .within_20_percent
                ).toFixed(
                  1
                )}%`
              : "--"
          }
          subtitle="Holdout result"
        />

      </div>

    </div>
  );
}


// =========================================================
// APPLICATION
// =========================================================

export default function App() {

  const [
    activePage,
    setActivePage,
  ] = useState(
    "Dashboard"
  );


  const [
    selectedSymbol,
    setSelectedSymbol,
  ] = useState(
    "RELIANCE.NS"
  );


  const [
    allStocks,
    setAllStocks,
  ] = useState(
    STOCKS
  );


  const [
    stockUniverseSource,
    setStockUniverseSource,
  ] = useState(
    "StockVision default"
  );


  const [
    stockUniverseLoading,
    setStockUniverseLoading,
  ] = useState(
    true
  );


  const [
    watchlistSymbols,
    setWatchlistSymbols,
  ] = useState(
    () => {

      try {

        const saved =
          localStorage.getItem(
            "stockvision_watchlist"
          );


        const parsed =
          saved
            ? JSON.parse(
                saved
              )
            : null;


        return Array.isArray(
          parsed
        )
          ? parsed
          : DEFAULT_WATCHLIST_SYMBOLS;

      } catch {

        return DEFAULT_WATCHLIST_SYMBOLS;
      }
    }
  );


  const [
    watchlistLiveData,
    setWatchlistLiveData,
  ] = useState(
    {}
  );


  const [
    watchlistLoading,
    setWatchlistLoading,
  ] = useState(
    false
  );


  const [
    selectedRange,
    setSelectedRange,
  ] = useState(
    () => {

      try {

        const saved =
          localStorage.getItem(
            "stockvision_chart_range"
          );

        return RANGE_OPTIONS.some(
          (
            item
          ) =>
            item.key ===
            saved
        )
          ? saved
          : "3mo";

      } catch {

        return "3mo";
      }
    }
  );


  const [
    stock,
    setStock,
  ] = useState(
    null
  );


  const [
    candleData,
    setCandleData,
  ] = useState(
    []
  );


  const [
    candleLoading,
    setCandleLoading,
  ] = useState(
    false
  );


  const [
    candleError,
    setCandleError,
  ] = useState(
    ""
  );


  const [
    prediction,
    setPrediction,
  ] = useState(
    null
  );


  const [
    relativePrediction,
    setRelativePrediction,
  ] = useState(
    null
  );


  const [
    futureForecast,
    setFutureForecast,
  ] = useState(
    null
  );


  const [
    futureStatus,
    setFutureStatus,
  ] = useState(
    null
  );


  const [
    analytics,
    setAnalytics,
  ] = useState(
    null
  );


  const [
    performanceData,
    setPerformanceData,
  ] = useState(
    {}
  );


  const [
    performanceLoading,
    setPerformanceLoading,
  ] = useState(
    false
  );


  const [
    marketOverview,
    setMarketOverview,
  ] = useState(
    {}
  );


  const [
    marketOverviewLoading,
    setMarketOverviewLoading,
  ] = useState(
    false
  );


  const [
    trackedStocks,
    setTrackedStocks,
  ] = useState(
    []
  );


  const [
    trackedStocksLoading,
    setTrackedStocksLoading,
  ] = useState(
    false
  );


  const [
    predictionHistory,
    setPredictionHistory,
  ] = useState(
    () => {

      try {

        const saved =
          localStorage.getItem(
            "stockvision_prediction_history"
          );


        return saved
          ? JSON.parse(
              saved
            )
          : [];

      } catch {

        return [];
      }
    }
  );


  const [
    search,
    setSearch,
  ] = useState(
    ""
  );


  const [
    searchOpen,
    setSearchOpen,
  ] = useState(
    false
  );


  const searchContainerRef =
    useRef(
      null
    );


  const [
    notificationOpen,
    setNotificationOpen,
  ] = useState(
    false
  );


  const [
    profileOpen,
    setProfileOpen,
  ] = useState(
    false
  );


  const [
    notificationsRead,
    setNotificationsRead,
  ] = useState(
    false
  );


  const [
    theme,
    setTheme,
  ] = useState(
    () => {

      try {

        return (
          localStorage.getItem(
            "stockvision_theme"
          ) ||
          "dark"
        );

      } catch {

        return "dark";
      }
    }
  );


  const [
    notificationsEnabled,
    setNotificationsEnabled,
  ] = useState(
    () => {

      try {

        const saved =
          localStorage.getItem(
            "stockvision_notifications_enabled"
          );

        return saved ===
          null
          ? true
          : saved ===
            "true";

      } catch {

        return true;
      }
    }
  );


  const [
    autoRefreshEnabled,
    setAutoRefreshEnabled,
  ] = useState(
    () => {

      try {

        const saved =
          localStorage.getItem(
            "stockvision_auto_refresh"
          );

        return saved ===
          null
          ? true
          : saved ===
            "true";

      } catch {

        return true;
      }
    }
  );


  const [
    refreshIntervalMs,
    setRefreshIntervalMs,
  ] = useState(
    () => {

      try {

        const saved =
          Number(
            localStorage.getItem(
              "stockvision_refresh_interval_ms"
            )
          );

        return [
          60000,
          120000,
          300000,
        ].includes(
          saved
        )
          ? saved
          : 60000;

      } catch {

        return 60000;
      }
    }
  );


  const [
    stockLoading,
    setStockLoading,
  ] = useState(
    false
  );


  const [
    predictionLoading,
    setPredictionLoading,
  ] = useState(
    false
  );


  const [
    relativeLoading,
    setRelativeLoading,
  ] = useState(
    false
  );


  const [
    futureLoading,
    setFutureLoading,
  ] = useState(
    false
  );


  const [
    analyticsLoading,
    setAnalyticsLoading,
  ] = useState(
    false
  );


  const [
    stockError,
    setStockError,
  ] = useState(
    ""
  );


  const [
    predictionError,
    setPredictionError,
  ] = useState(
    ""
  );


  const [
    relativeError,
    setRelativeError,
  ] = useState(
    ""
  );


  const [
    futureError,
    setFutureError,
  ] = useState(
    ""
  );


  const [
    analyticsError,
    setAnalyticsError,
  ] = useState(
    ""
  );


  // =======================================================
  // LOAD CURRENT NSE STOCK UNIVERSE
  // =======================================================

  useEffect(
    () => {

      let cancelled =
        false;


      async function loadStockUniverse() {

        try {

          setStockUniverseLoading(
            true
          );


          const response =
            await fetch(
              `${API_URL}/stocks`
            );


          const data =
            await response
              .json()
              .catch(
                () =>
                  null
              );


          if (
            !response.ok
          ) {

            throw new Error(
              data?.detail ||
              "Unable to load NSE stock list."
            );
          }


          const stocks =
            Array.isArray(
              data?.stocks
            )
              ? data.stocks.filter(
                  (
                    item
                  ) =>
                    item?.symbol &&
                    item?.short
                )
              : [];


          if (
            !cancelled &&
            stocks.length >
              0
          ) {

            setAllStocks(
              stocks
            );

            setStockUniverseSource(
              data?.source ||
              "NSE"
            );
          }


        } catch {

          if (
            !cancelled
          ) {

            setAllStocks(
              STOCKS
            );

            setStockUniverseSource(
              "StockVision fallback"
            );
          }


        } finally {

          if (
            !cancelled
          ) {

            setStockUniverseLoading(
              false
            );
          }
        }
      }


      loadStockUniverse();


      return () => {

        cancelled =
          true;
      };

    },
    []
  );


  // =======================================================
  // SELECTED STOCK INFO
  // =======================================================

  const selectedStockInfo =
    allStocks.find(
      (item) =>
        item.symbol ===
        selectedSymbol
    ) || {

      name:
        selectedSymbol.replace(
          ".NS",
          ""
        ),

      symbol:
        selectedSymbol,

      short:
        selectedSymbol.replace(
          ".NS",
          ""
        ),

    };


  // =======================================================
  // DYNAMIC WATCHLIST
  // =======================================================

  const watchlistItems =
    useMemo(
      () => {

        return watchlistSymbols.map(
          (
            symbol
          ) => {

            return (
              allStocks.find(
                (
                  item
                ) =>
                  item.symbol ===
                  symbol
              ) || {
                symbol,
                short:
                  symbol.replace(
                    ".NS",
                    ""
                  ),
                name:
                  symbol.replace(
                    ".NS",
                    ""
                  ),
              }
            );
          }
        );

      },
      [
        watchlistSymbols,
        allStocks,
      ]
    );


  function isWatchlisted(
    symbol
  ) {

    return watchlistSymbols.includes(
      symbol
    );
  }


  function toggleWatchlist(
    itemOrSymbol
  ) {

    const symbol =
      typeof itemOrSymbol ===
      "string"
        ? itemOrSymbol
        : itemOrSymbol?.symbol;


    if (!symbol) {
      return;
    }


    setWatchlistSymbols(
      (
        current
      ) => {

        if (
          current.includes(
            symbol
          )
        ) {

          return current.filter(
            (
              item
            ) =>
              item !==
              symbol
          );
        }


        return [
          ...current,
          symbol,
        ];
      }
    );
  }


  useEffect(
    () => {

      try {

        localStorage.setItem(
          "stockvision_watchlist",
          JSON.stringify(
            watchlistSymbols
          )
        );

      } catch {
        // Ignore storage errors.
      }

    },
    [
      watchlistSymbols,
    ]
  );


  useEffect(
    () => {

      let cancelled =
        false;


      async function fetchWatchlistPrices() {

        if (
          watchlistSymbols.length ===
          0
        ) {

          setWatchlistLiveData(
            {}
          );

          return;
        }


        try {

          setWatchlistLoading(
            true
          );


          const results =
            await Promise.allSettled(
              watchlistSymbols.map(
                async (
                  symbol
                ) => {

                  const response =
                    await fetch(
                      `${API_URL}/stock/${encodeURIComponent(
                        symbol
                      )}?range=1d`
                    );


                  const data =
                    await response
                      .json()
                      .catch(
                        () =>
                          null
                      );


                  if (
                    !response.ok
                  ) {

                    throw new Error(
                      data?.detail ||
                      `Unable to load ${symbol}`
                    );
                  }


                  return {
                    symbol,
                    price:
                      Number(
                        data?.price
                      ),
                    change:
                      Number(
                        data?.change
                      ),
                    change_percent:
                      Number(
                        data?.change_percent
                      ),
                    volume:
                      Number(
                        data?.volume
                      ),
                  };
                }
              )
            );


          const next = {};


          results.forEach(
            (
              result
            ) => {

              if (
                result.status ===
                "fulfilled"
              ) {

                next[
                  result.value.symbol
                ] =
                  result.value;
              }
            }
          );


          if (
            !cancelled
          ) {

            setWatchlistLiveData(
              next
            );
          }


        } finally {

          if (
            !cancelled
          ) {

            setWatchlistLoading(
              false
            );
          }
        }
      }


      fetchWatchlistPrices();


      const refreshId =
        autoRefreshEnabled
          ? window.setInterval(
              fetchWatchlistPrices,
              refreshIntervalMs
            )
          : null;


      return () => {

        cancelled =
          true;

        if (
          refreshId
        ) {

          window.clearInterval(
            refreshId
          );
        }
      };

    },
    [
      watchlistSymbols,
      autoRefreshEnabled,
      refreshIntervalMs,
    ]
  );


  // =======================================================
  // SEARCH RESULTS
  // =======================================================

  const filteredStocks =
    useMemo(
      () => {

        const query =
          search
            .trim()
            .toLowerCase();


        if (!query) {

          return allStocks.slice(
            0,
            7
          );
        }


        return allStocks.filter(
          (item) =>

            item.name
              .toLowerCase()
              .includes(
                query
              )

            ||

            item.short
              .toLowerCase()
              .includes(
                query
              )

            ||

            item.symbol
              .toLowerCase()
              .includes(
                query
              )

        ).slice(
          0,
          8
        );

      },
      [
        search,
        allStocks,
      ]
    );


  // =======================================================
  // CLOSE SEARCH WHEN CLICKING OUTSIDE
  // =======================================================

  useEffect(
    () => {

      function handleOutsideSearchClick(
        event
      ) {

        if (
          searchOpen &&
          searchContainerRef.current &&
          !searchContainerRef.current.contains(
            event.target
          )
        ) {

          setSearchOpen(
            false
          );
        }
      }


      document.addEventListener(
        "mousedown",
        handleOutsideSearchClick
      );


      document.addEventListener(
        "touchstart",
        handleOutsideSearchClick
      );


      return () => {

        document.removeEventListener(
          "mousedown",
          handleOutsideSearchClick
        );


        document.removeEventListener(
          "touchstart",
          handleOutsideSearchClick
        );
      };

    },
    [
      searchOpen,
    ]
  );


  // =======================================================
  // HEADER NOTIFICATIONS
  // =======================================================

  const headerNotifications =
    useMemo(
      () => {

        const items = [];


        if (stock) {

          items.push({
            id:
              "market",
            type:
              "market",
            title:
              "Market Data Updated",
            message:
              `${selectedStockInfo.short} is ${formatPrice(
                stock.price
              )} (${formatPercent(
                stock.change_percent
              )}).`,
          });
        }


        if (prediction) {

          items.push({
            id:
              "prediction",
            type:
              "ai",
            title:
              "AI Prediction Ready",
            message:
              `${selectedStockInfo.short} X2 forecast: ${formatPercent(
                prediction?.experimental_x2_point?.move_percent ??
                prediction.predicted_return_percent,
                4
              )}.`,
          });
        }


        if (
          relativePrediction?.signal
        ) {

          items.push({
            id:
              "v9",
            type:
              "v9",
            title:
              "V9 Signal Updated",
            message:
              `${selectedStockInfo.short}: ${String(
                relativePrediction.signal
              ).toUpperCase()} vs NIFTY 50.`,
          });
        }


        if (
          futureStatus?.status ===
          "ready"
        ) {

          items.push({
            id:
              "forecast",
            type:
              "forecast",
            title:
              "Future Forecast Ready",
            message:
              `1D / 3D / 5D / 10D forecast is ready for ${selectedStockInfo.short}.`,
          });
        }


        if (
          futureStatus?.status ===
          "training"
        ) {

          items.push({
            id:
              "training",
            type:
              "forecast",
            title:
              "Forecast Model Training",
            message:
              `${selectedStockInfo.short} multi-horizon model is being prepared.`,
          });
        }


        if (
          stockError ||
          predictionError ||
          relativeError
        ) {

          items.push({
            id:
              "warning",
            type:
              "warning",
            title:
              "Service Notice",
            message:
              stockError ||
              predictionError ||
              relativeError,
          });
        }


        return items.slice(
          0,
          6
        );

      },
      [
        stock,
        prediction,
        relativePrediction,
        futureStatus,
        stockError,
        predictionError,
        relativeError,
        selectedStockInfo.short,
      ]
    );


  const unreadNotificationCount =
    !notificationsEnabled ||
    notificationsRead
      ? 0
      : headerNotifications.length;


  const isLightTheme =
    theme ===
    "light";


  // =======================================================
  // FETCH STOCK
  // =======================================================

  async function fetchStock(
    symbol,
    range = selectedRange
  ) {

    try {

      setStockLoading(
        true
      );

      setStockError(
        ""
      );


      const response =
        await fetch(

          `${API_URL}/stock/${encodeURIComponent(
            symbol
          )}?range=${encodeURIComponent(
            range
          )}`

        );


      const data =
        await response
          .json()
          .catch(
            () =>
              null
          );


      if (
        !response.ok
      ) {

        throw new Error(
          data?.detail ||
          `Unable to load ${symbol}`
        );
      }


      setStock(
        data
      );


    } catch (
      error
    ) {

      setStockError(
        error.message ||
        "Unable to load market data."
      );


    } finally {

      setStockLoading(
        false
      );
    }
  }


  // =======================================================
  // FETCH REAL OHLC CANDLESTICK DATA
  // =======================================================

  async function fetchCandles(
    symbol,
    range = selectedRange
  ) {

    const config =
      CANDLE_RANGE_CONFIG[
        range
      ] ||
      CANDLE_RANGE_CONFIG["1d"];


    try {

      setCandleLoading(
        true
      );

      setCandleError(
        ""
      );


      const response =
        await fetch(

          `${API_URL}/stock-candles/${encodeURIComponent(
            symbol
          )}?period=${encodeURIComponent(
            config.period
          )}&interval=${encodeURIComponent(
            config.interval
          )}`

        );


      const data =
        await response
          .json()
          .catch(
            () =>
              null
          );


      if (
        !response.ok
      ) {

        throw new Error(
          data?.detail ||
          "Candlestick data unavailable."
        );
      }


      setCandleData(
        Array.isArray(
          data?.candles
        )
          ? data.candles
          : []
      );


    } catch (
      error
    ) {

      setCandleData(
        []
      );


      setCandleError(
        error.message ||
        "Candlestick data unavailable."
      );


    } finally {

      setCandleLoading(
        false
      );
    }
  }


  // =======================================================
  // FETCH NEXT DAY PREDICTION
  // =======================================================

  async function fetchPrediction(
    symbol
  ) {

    try {

      setPredictionLoading(
        true
      );

      setPredictionError(
        ""
      );


      const response =
        await fetch(

          `${API_URL}/predict/${encodeURIComponent(
            symbol
          )}`

        );


      const data =
        await response
          .json()
          .catch(
            () =>
              null
          );


      if (
        !response.ok
      ) {

        throw new Error(
          data?.detail ||
          "Prediction unavailable."
        );
      }


      setPrediction(
        data
      );


    } catch (
      error
    ) {

      setPrediction(
        null
      );


      setPredictionError(
        error.message ||
        "Prediction unavailable."
      );


    } finally {

      setPredictionLoading(
        false
      );
    }
  }


  // =======================================================
  // FETCH V9 RELATIVE-STRENGTH PREDICTION
  // =======================================================

  async function fetchRelativePrediction(
    symbol
  ) {

    try {

      setRelativeLoading(
        true
      );

      setRelativeError(
        ""
      );


      const response =
        await fetch(

          `${API_URL}/relative-predict/${encodeURIComponent(
            symbol
          )}`

        );


      const data =
        await response
          .json()
          .catch(
            () =>
              null
          );


      if (
        !response.ok
      ) {

        throw new Error(
          data?.detail ||
          "Relative-strength prediction unavailable."
        );
      }


      setRelativePrediction(
        data
      );


    } catch (
      error
    ) {

      setRelativePrediction(
        null
      );


      setRelativeError(
        error.message ||
        "Relative-strength prediction unavailable."
      );


    } finally {

      setRelativeLoading(
        false
      );
    }
  }


  // =======================================================
  // FETCH MODEL ANALYTICS
  // =======================================================

  async function fetchModelAnalytics() {

    try {

      setAnalyticsLoading(
        true
      );

      setAnalyticsError(
        ""
      );


      const response =
        await fetch(
          `${API_URL}/model-analytics`
        );


      const data =
        await response
          .json()
          .catch(
            () =>
              null
          );


      if (
        !response.ok
      ) {

        throw new Error(
          data?.detail ||
          "Model analytics unavailable."
        );
      }


      setAnalytics(
        data
      );


    } catch (
      error
    ) {

      setAnalyticsError(
        error.message ||
        "Model analytics unavailable."
      );


    } finally {

      setAnalyticsLoading(
        false
      );
    }
  }


  // =======================================================
  // FETCH MARKET OVERVIEW
  // =======================================================

  async function fetchMarketOverview() {

    const marketSymbols = [
      {
        key: "nifty",
        symbol: "^NSEI",
      },
      {
        key: "sensex",
        symbol: "^BSESN",
      },
      {
        key: "bankNifty",
        symbol: "^NSEBANK",
      },
    ];


    try {

      setMarketOverviewLoading(
        true
      );


      const results =
        await Promise.allSettled(

          marketSymbols.map(
            async (
              item
            ) => {

              const response =
                await fetch(

                  `${API_URL}/stock/${encodeURIComponent(
                    item.symbol
                  )}?range=1d`

                );


              const data =
                await response
                  .json()
                  .catch(
                    () =>
                      null
                  );


              if (
                !response.ok
              ) {

                throw new Error(
                  data?.detail ||
                  `Unable to load ${item.symbol}`
                );
              }


              return {
                key:
                  item.key,
                data,
              };
            }
          )

        );


      const nextOverview =
        {};


      results.forEach(
        (
          result
        ) => {

          if (
            result.status ===
            "fulfilled"
          ) {

            nextOverview[
              result.value.key
            ] =
              result.value.data;
          }

        }
      );


      setMarketOverview(
        (
          current
        ) => ({
          ...current,
          ...nextOverview,
        })
      );


    } catch (
      error
    ) {

      console.error(
        "Market overview error:",
        error
      );


    } finally {

      setMarketOverviewLoading(
        false
      );
    }
  }


  // =======================================================
  // FETCH TRACKED STOCKS / MARKET MOVERS
  // =======================================================

  async function fetchTrackedStocks() {

    try {

      setTrackedStocksLoading(
        true
      );


      const results =
        await Promise.allSettled(

          STOCKS.slice(0, 12).map(
            async (
              item
            ) => {

              const response =
                await fetch(
                  `${API_URL}/stock/${encodeURIComponent(
                    item.symbol
                  )}?range=1d`
                );


              const data =
                await response
                  .json()
                  .catch(
                    () =>
                      null
                  );


              if (
                !response.ok
              ) {
                throw new Error(
                  data?.detail ||
                  `Unable to load ${item.symbol}`
                );
              }


              return {
                ...item,
                price:
                  Number(
                    data?.price
                  ),
                change:
                  Number(
                    data?.change
                  ),
                change_percent:
                  Number(
                    data?.change_percent
                  ),
              };
            }
          )
        );


      const next =
        results
          .filter(
            (result) =>
              result.status ===
              "fulfilled"
          )
          .map(
            (result) =>
              result.value
          );


      setTrackedStocks(
        next
      );


    } catch (
      error
    ) {

      console.error(
        "Tracked stocks error:",
        error
      );


    } finally {

      setTrackedStocksLoading(
        false
      );
    }
  }


  // =======================================================
  // CHECK FUTURE MODEL STATUS
  // =======================================================

  async function checkFutureStatus(
    symbol
  ) {

    const response =
      await fetch(

        `${API_URL}/future-status/${encodeURIComponent(
          symbol
        )}`

      );


    const data =
      await response
        .json()
        .catch(
          () =>
            null
        );


    if (
      !response.ok
    ) {

      throw new Error(
        data?.detail ||
        "Unable to check forecast model status."
      );
    }


    setFutureStatus(
      data
    );


    return data;
  }


  // =======================================================
  // START FUTURE TRAINING
  // =======================================================

  async function startFutureTraining(
    symbol
  ) {

    try {

      setFutureError(
        ""
      );


      const response =
        await fetch(

          `${API_URL}/future-train/${encodeURIComponent(
            symbol
          )}`,

          {
            method:
              "POST",
          }

        );


      const data =
        await response
          .json()
          .catch(
            () =>
              null
          );


      if (
        !response.ok
      ) {

        throw new Error(
          data?.detail ||
          "Unable to start model training."
        );
      }


      setFutureStatus(
        data
      );


      return data;


    } catch (
      error
    ) {

      setFutureError(
        error.message ||
        "Unable to start model training."
      );


      return null;
    }
  }


  // =======================================================
  // FETCH FUTURE FORECAST
  // =======================================================

  async function fetchFutureForecast(
    symbol
  ) {

    try {

      setFutureLoading(
        true
      );

      setFutureError(
        ""
      );


      const response =
        await fetch(

          `${API_URL}/future-predict/${encodeURIComponent(
            symbol
          )}`

        );


      const data =
        await response
          .json()
          .catch(
            () =>
              null
          );


      if (
        !response.ok
      ) {

        throw new Error(
          data?.detail ||
          "Future forecast unavailable."
        );
      }


      setFutureForecast(
        data
      );


    } catch (
      error
    ) {

      setFutureForecast(
        null
      );


      setFutureError(
        error.message ||
        "Future forecast unavailable."
      );


    } finally {

      setFutureLoading(
        false
      );
    }
  }


  // =======================================================
  // PREPARE FUTURE FORECAST
  // =======================================================

  async function prepareFutureForecast(
    symbol
  ) {

    try {

      setFutureError(
        ""
      );


      setFutureForecast(
        null
      );


      const status =
        await checkFutureStatus(
          symbol
        );


      // MODEL READY

      if (
        status.status ===
        "ready"
      ) {

        await fetchFutureForecast(
          symbol
        );

        return;
      }


      // MODEL DOESN'T EXIST

      if (
        status.status ===
        "idle"
      ) {

        await startFutureTraining(
          symbol
        );

        return;
      }


      // ALREADY TRAINING

      if (
        status.status ===
        "training"
      ) {

        return;
      }


      // ERROR

      if (
        status.status ===
        "error"
      ) {

        setFutureError(
          status.error ||
          "Model training failed."
        );
      }


    } catch (
      error
    ) {

      setFutureError(
        error.message ||
        "Unable to prepare future forecast."
      );
    }
  }


  // =======================================================
  // MARKET OVERVIEW EFFECT
  // =======================================================

  useEffect(
    () => {

      fetchMarketOverview();
      fetchTrackedStocks();


      if (
        !autoRefreshEnabled
      ) {

        return;
      }


      const interval =
        setInterval(
          () => {

            fetchMarketOverview();
            fetchTrackedStocks();

          },
          refreshIntervalMs
        );


      return () => {

        clearInterval(
          interval
        );
      };

    },
    [
      autoRefreshEnabled,
      refreshIntervalMs,
    ]
  );


  // =======================================================
  // FETCH STOCK PERFORMANCE SNAPSHOT
  // =======================================================

  async function fetchPerformanceSnapshot(
    symbol
  ) {

    const ranges = [
      {
        key: "1D",
        range: "1d",
      },
      {
        key: "1W",
        range: "5d",
      },
      {
        key: "1M",
        range: "1mo",
      },
      {
        key: "6M",
        range: "6mo",
      },
      {
        key: "1Y",
        range: "1y",
      },
    ];


    try {

      setPerformanceLoading(
        true
      );


      const results =
        await Promise.all(
          ranges.map(
            async (
              item
            ) => {

              const response =
                await fetch(
                  `${API_URL}/stock/${encodeURIComponent(
                    symbol
                  )}?range=${encodeURIComponent(
                    item.range
                  )}`
                );


              if (
                !response.ok
              ) {
                return [
                  item.key,
                  null,
                ];
              }


              const data =
                await response.json();


              const chart =
                Array.isArray(
                  data?.chart
                )
                  ? data.chart
                  : [];


              let move = null;


              if (
                chart.length >= 2
              ) {

                const firstPrice =
                  Number(
                    chart[0]?.price
                  );


                const lastPrice =
                  Number(
                    chart[
                      chart.length -
                      1
                    ]?.price
                  );


                if (
                  Number.isFinite(
                    firstPrice
                  ) &&
                  Number.isFinite(
                    lastPrice
                  ) &&
                  firstPrice !== 0
                ) {

                  move =
                    (
                      (
                        lastPrice -
                        firstPrice
                      ) /
                      firstPrice
                    ) *
                    100;
                }
              }


              if (
                item.key === "1D" &&
                data?.change_percent !==
                  undefined
              ) {

                move =
                  Number(
                    data.change_percent
                  );
              }


              return [
                item.key,
                move,
              ];

            }
          )
        );


      setPerformanceData(
        Object.fromEntries(
          results
        )
      );


    } catch (
      error
    ) {

      console.error(
        "Performance snapshot error:",
        error
      );


    } finally {

      setPerformanceLoading(
        false
      );
    }
  }


  // =======================================================
  // THEME PREFERENCE
  // =======================================================

  useEffect(
    () => {

      try {

        localStorage.setItem(
          "stockvision_theme",
          theme
        );

      } catch {
        // Ignore localStorage errors.
      }

    },
    [
      theme,
    ]
  );


  // =======================================================
  // APP SETTINGS PREFERENCES
  // =======================================================

  useEffect(
    () => {

      try {

        localStorage.setItem(
          "stockvision_notifications_enabled",
          String(
            notificationsEnabled
          )
        );

        localStorage.setItem(
          "stockvision_auto_refresh",
          String(
            autoRefreshEnabled
          )
        );

        localStorage.setItem(
          "stockvision_refresh_interval_ms",
          String(
            refreshIntervalMs
          )
        );

        localStorage.setItem(
          "stockvision_chart_range",
          selectedRange
        );

      } catch {
        // Ignore localStorage errors.
      }

    },
    [
      notificationsEnabled,
      autoRefreshEnabled,
      refreshIntervalMs,
      selectedRange,
    ]
  );


  useEffect(
    () => {

      if (
        !notificationsEnabled
      ) {

        setNotificationOpen(
          false
        );

        setNotificationsRead(
          true
        );
      }

    },
    [
      notificationsEnabled,
    ]
  );


  // =======================================================
  // MARKET DATA EFFECT
  // =======================================================

  useEffect(
    () => {

      fetchStock(
        selectedSymbol,
        selectedRange
      );


      fetchCandles(
        selectedSymbol,
        selectedRange
      );

    },
    [
      selectedSymbol,
      selectedRange,
    ]
  );


  // =======================================================
  // NEXT DAY MODEL EFFECT
  // =======================================================

  useEffect(
    () => {

      fetchPrediction(
        selectedSymbol
      );

    },
    [
      selectedSymbol,
    ]
  );


  // =======================================================
  // V9 RELATIVE-STRENGTH EFFECT
  // =======================================================

  useEffect(
    () => {

      fetchRelativePrediction(
        selectedSymbol
      );

    },
    [
      selectedSymbol,
    ]
  );


  // =======================================================
  // STOCK PERFORMANCE EFFECT
  // =======================================================

  useEffect(
    () => {

      fetchPerformanceSnapshot(
        selectedSymbol
      );

    },
    [
      selectedSymbol,
    ]
  );


  // =======================================================
  // LOCAL PREDICTION HISTORY
  // =======================================================

  useEffect(
    () => {

      if (
        !relativePrediction?.signal
      ) {
        return;
      }


      const today =
        new Date()
          .toLocaleDateString(
            "en-GB",
            {
              day:
                "2-digit",
              month:
                "short",
              year:
                "numeric",
            }
          );


      const entry = {
        key:
          `${selectedSymbol}-${today}`,
        date:
          today,
        stock:
          selectedSymbol.replace(
            ".NS",
            ""
          ),
        signal:
          String(
            relativePrediction.signal
          ).toUpperCase(),
        score:
          Number(
            relativePrediction.top_probability ||
            0
          ),
        outcome:
          "Pending",
      };


      setPredictionHistory(
        (
          previous
        ) => {

          const withoutToday =
            previous.filter(
              (
                item
              ) =>
                item.key !==
                entry.key
            );


          const next = [
            entry,
            ...withoutToday,
          ].slice(
            0,
            8
          );


          try {

            localStorage.setItem(
              "stockvision_prediction_history",
              JSON.stringify(
                next
              )
            );

          } catch {
            // Ignore browser storage failures.
          }


          return next;
        }
      );

    },
    [
      relativePrediction,
      selectedSymbol,
    ]
  );


  // =======================================================
  // MODEL ANALYTICS PAGE EFFECT
  // =======================================================

  useEffect(
    () => {

      if (
        activePage !==
        "Model Analytics"
      ) {
        return;
      }


      if (
        !analytics
      ) {
        fetchModelAnalytics();
      }

    },
    [
      activePage,
    ]
  );


  // =======================================================
  // FUTURE FORECAST PAGE EFFECT
  // =======================================================

  useEffect(
    () => {

      if (
        activePage !==
        "Future Forecast"
      ) {

        return;
      }


      setFutureStatus(
        null
      );

      setFutureForecast(
        null
      );

      setFutureError(
        ""
      );


      prepareFutureForecast(
        selectedSymbol
      );

    },
    [
      activePage,
      selectedSymbol,
    ]
  );


  // =======================================================
  // TRAINING STATUS POLLING
  // =======================================================

  useEffect(
    () => {

      if (
        activePage !==
        "Future Forecast"
      ) {

        return;
      }


      if (
        futureStatus?.status !==
        "training"
      ) {

        return;
      }


      const interval =
        setInterval(
          async () => {

            try {

              const status =
                await checkFutureStatus(
                  selectedSymbol
                );


              if (
                status.status ===
                "ready"
              ) {

                clearInterval(
                  interval
                );


                await fetchFutureForecast(
                  selectedSymbol
                );
              }


              if (
                status.status ===
                "error"
              ) {

                clearInterval(
                  interval
                );


                setFutureError(
                  status.error ||
                  "Model training failed."
                );
              }


            } catch (
              error
            ) {

              console.error(
                error
              );
            }

          },
          2500
        );


      return () => {

        clearInterval(
          interval
        );
      };

    },
    [
      activePage,
      selectedSymbol,
      futureStatus?.status,
    ]
  );


  // =======================================================
  // MARKET AUTO REFRESH
  // =======================================================

  useEffect(
    () => {

      if (
        !autoRefreshEnabled
      ) {

        return;
      }


      const interval =
        setInterval(
          () => {

            fetchStock(
              selectedSymbol,
              selectedRange
            );


            fetchCandles(
              selectedSymbol,
              selectedRange
            );

          },
          refreshIntervalMs
        );


      return () => {

        clearInterval(
          interval
        );
      };

    },
    [
      selectedSymbol,
      selectedRange,
      autoRefreshEnabled,
      refreshIntervalMs,
    ]
  );


  // =======================================================
  // SELECT STOCK
  // =======================================================

  function selectStock(
    symbol
  ) {

    const normalized =
      normalizeStockSymbol(
        symbol
      );


    if (
      !normalized
    ) {

      return;
    }


    setSelectedSymbol(
      normalized
    );


    setSearch(
      ""
    );


    setSearchOpen(
      false
    );


    setRelativePrediction(
      null
    );


    setRelativeError(
      ""
    );


    setCandleData(
      []
    );


    setCandleError(
      ""
    );


    setFutureForecast(
      null
    );


    setFutureStatus(
      null
    );


    setFutureError(
      ""
    );
  }


  // =======================================================
  // SEARCH ENTER
  // =======================================================

  function handleSearchEnter() {

    const cleanSearch =
      search
        .trim()
        .toLowerCase();


    if (
      !cleanSearch
    ) {

      return;
    }


    // EXACT MATCH

    const exactMatch =
      allStocks.find(
        (item) =>

          item.short
            .toLowerCase() ===
          cleanSearch

          ||

          item.symbol
            .toLowerCase() ===
          cleanSearch

          ||

          item.symbol
            .replace(
              ".NS",
              ""
            )
            .toLowerCase() ===
          cleanSearch

          ||

          item.name
            .toLowerCase() ===
          cleanSearch
      );


    if (
      exactMatch
    ) {

      selectStock(
        exactMatch.symbol
      );

      return;
    }


    // PARTIAL MATCH
    //
    // Example:
    // AXIS -> AXISBANK
    // WIP  -> WIPRO
    // REL  -> RELIANCE

    const partialMatch =
      allStocks.find(
        (item) =>

          item.short
            .toLowerCase()
            .includes(
              cleanSearch
            )

          ||

          item.name
            .toLowerCase()
            .includes(
              cleanSearch
            )

          ||

          item.symbol
            .toLowerCase()
            .includes(
              cleanSearch
            )
      );


    if (
      partialMatch
    ) {

      selectStock(
        partialMatch.symbol
      );

      return;
    }


    // CUSTOM NSE TICKER

    selectStock(
      cleanSearch
    );
  }


  // =======================================================
  // MARKET VALUES
  // =======================================================

  const currentPrice =
    Number(
      stock?.price ||
      0
    );


  const change =
    Number(
      stock?.change ||
      0
    );


  const changePercent =
    Number(
      stock?.change_percent ||
      0
    );


  const positiveDay =
    change >= 0;


  const indicators =
    stock?.indicators ||
    {};


  const rsiValue =
    indicators?.rsi14 ??
    indicators?.rsi ??
    stock?.rsi14 ??
    null;


  const rsiStatus =
    rsiValue === null ||
    rsiValue === undefined

      ? "No data"

      : Number(
          rsiValue
        ) >= 70

      ? "Overbought"

      : Number(
          rsiValue
        ) <= 30

      ? "Oversold"

      : "Neutral";


  const rawMacd =
    indicators?.macd;


  const macdValue =
    typeof rawMacd ===
      "object"

      ? rawMacd?.macd

      : rawMacd ??
        stock?.macd ??
        null;


  const macdStatus =
    macdValue === null ||
    macdValue === undefined

      ? "No data"

      : Number(
          macdValue
        ) >= 0

      ? "Bullish"

      : "Bearish";


  const sma20Value =
    indicators?.sma20 ??
    indicators?.sma_20 ??
    stock?.sma20 ??
    null;


  const ema20Value =
    indicators?.ema20 ??
    indicators?.ema_20 ??
    stock?.ema20 ??
    null;


  // =======================================================
  // DASHBOARD PAGE
  // =======================================================

  function DashboardPage() {

    const forecastMove =
      Number(
        prediction?.predicted_return_percent ||
        0
      );


    const forecastSignal =
      prediction?.trend_signal ||
      (
        forecastMove >= 0.25
          ? "BULLISH"
          : forecastMove <= -0.25
          ? "BEARISH"
          : "NEUTRAL"
      );


    const relativeSignal =
      String(
        relativePrediction?.signal ||
        "NEUTRAL"
      ).toUpperCase();


    const relativeContext =
      relativePrediction?.current_market_context ||
      {};


    const relative5d =
      Number(
        relativeContext.previous_5d_relative_strength_percent
      );


    const marketRows =
      [...trackedStocks].sort(
        (a, b) =>
          Number(
            b.change_percent ||
            0
          ) -
          Number(
            a.change_percent ||
            0
          )
      );


    const gainers =
      marketRows.slice(
        0,
        3
      );


    const losers =
      [...marketRows]
        .reverse()
        .slice(
          0,
          3
        );


    const whyItems = [];


    if (
      Number.isFinite(
        relative5d
      )
    ) {

      whyItems.push({
        positive:
          relative5d >= 0,
        text:
          relative5d >= 0
            ? `Positive 5-day relative strength vs NIFTY (${formatPercent(relative5d)}).`
            : `Negative 5-day relative strength vs NIFTY (${formatPercent(relative5d)}).`,
      });
    }


    if (
      sma20Value !== null &&
      sma20Value !== undefined
    ) {

      whyItems.push({
        positive:
          currentPrice >=
          Number(
            sma20Value
          ),
        text:
          currentPrice >=
          Number(
            sma20Value
          )
            ? "Price is trading above SMA 20."
            : "Price is trading below SMA 20.",
      });
    }


    if (
      ema20Value !== null &&
      ema20Value !== undefined
    ) {

      whyItems.push({
        positive:
          currentPrice >=
          Number(
            ema20Value
          ),
        text:
          currentPrice >=
          Number(
            ema20Value
          )
            ? "Price is above EMA 20, supporting short-term strength."
            : "Price is below EMA 20, showing short-term weakness.",
      });
    }


    if (
      macdValue !== null &&
      macdValue !== undefined
    ) {

      whyItems.push({
        positive:
          Number(
            macdValue
          ) >= 0,
        text:
          Number(
            macdValue
          ) >= 0
            ? "MACD is positive."
            : "MACD is negative.",
      });
    }


    if (
      rsiValue !== null &&
      rsiValue !== undefined
    ) {

      const rsi =
        Number(
          rsiValue
        );

      whyItems.push({
        positive:
          rsi >= 40 &&
          rsi <= 70,
        text:
          rsi >= 70
            ? "RSI is in an overbought zone."
            : rsi <= 30
            ? "RSI is in an oversold zone."
            : "RSI remains in a neutral zone.",
      });
    }


    const performanceCards = [
      "1D",
      "1W",
      "1M",
      "6M",
      "1Y",
    ];


    const indicatorItems = [
      {
        label: "RSI (14)",
        value:
          rsiValue !== null &&
          rsiValue !== undefined
            ? Number(
                rsiValue
              ).toFixed(
                2
              )
            : "--",
        status:
          rsiStatus,
        positive:
          rsiStatus !==
            "Overbought",
      },
      {
        label: "MACD",
        value:
          macdValue !== null &&
          macdValue !== undefined
            ? Number(
                macdValue
              ).toFixed(
                2
              )
            : "--",
        status:
          macdStatus,
        positive:
          macdStatus ===
            "Bullish",
      },
      {
        label: "SMA (20)",
        value:
          formatPrice(
            sma20Value
          ),
        status:
          sma20Value
            ? currentPrice >=
              Number(
                sma20Value
              )
              ? "Price > SMA"
              : "Price < SMA"
            : "--",
        positive:
          sma20Value &&
          currentPrice >=
            Number(
              sma20Value
            ),
      },
      {
        label: "EMA (20)",
        value:
          formatPrice(
            ema20Value
          ),
        status:
          ema20Value
            ? currentPrice >=
              Number(
                ema20Value
              )
              ? "Price > EMA"
              : "Price < EMA"
            : "--",
        positive:
          ema20Value &&
          currentPrice >=
            Number(
              ema20Value
            ),
      },
    ];


    return (
      <div className="space-y-2.5">

        {/* =================================================
            TOP MARKET STRIP — 5 CARDS LIKE THE REFERENCE
        ================================================= */}

        <section className="grid gap-2.5 md:grid-cols-2 xl:grid-cols-5">

          <MarketOverviewCard
            title="NIFTY 50"
            symbol="^NSEI"
            data={
              marketOverview.nifty
            }
            loading={
              marketOverviewLoading
            }
          />

          <MarketOverviewCard
            title="SENSEX"
            symbol="^BSESN"
            data={
              marketOverview.sensex
            }
            loading={
              marketOverviewLoading
            }
          />

          <MarketOverviewCard
            title="BANK NIFTY"
            symbol="^NSEBANK"
            data={
              marketOverview.bankNifty
            }
            loading={
              marketOverviewLoading
            }
          />

          <MarketStatusCard
            data={
              marketOverview
            }
            loading={
              marketOverviewLoading
            }
          />

          <MarketBreadthCard
            items={
              trackedStocks
            }
            loading={
              trackedStocksLoading
            }
          />

        </section>


        {stockError && (
          <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-400">
            {stockError}
          </div>
        )}


        {/* =================================================
            PRIMARY WORKSPACE
            LEFT: PRICE CHART
            RIGHT: AI + V9
        ================================================= */}

        <section className="grid gap-3 xl:grid-cols-[minmax(0,1.62fr)_minmax(340px,0.78fr)]">

          {/* MARKET CHART */}

          <div className="overflow-hidden rounded-xl border border-[#1b2738] bg-[#0f141d]">

            <div className="border-b border-white/5 px-4 py-4">

              <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-start">

                <div>

                  <div className="flex flex-wrap items-center gap-2">

                    <h1 className="text-base font-semibold text-white">
                      {selectedStockInfo.name.toUpperCase()} · {RANGE_OPTIONS.find(
                        (item) =>
                          item.key ===
                          selectedRange
                      )?.label || "1D"} · NSE
                    </h1>

                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                        positiveDay
                          ? "bg-green-500/10 text-green-400"
                          : "bg-red-500/10 text-red-400"
                      }`}
                    >
                      {formatPercent(
                        changePercent
                      )}
                    </span>

                  </div>


                  <div className="mt-2 flex flex-wrap items-end gap-x-3 gap-y-1">

                    <p className="text-3xl font-bold tracking-tight text-white">
                      {formatPrice(
                        currentPrice
                      )}
                    </p>

                    <p
                      className={`pb-1 text-sm font-semibold ${
                        positiveDay
                          ? "text-green-400"
                          : "text-red-400"
                      }`}
                    >
                      {change >= 0
                        ? "+"
                        : ""}
                      {change.toFixed(
                        2
                      )}{" "}
                      ({formatPercent(
                        changePercent
                      )})
                    </p>

                  </div>


                  <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-gray-600">

                    <span>
                      O{" "}
                      <b className="font-medium text-gray-400">
                        {formatPrice(
                          stock?.open
                        )}
                      </b>
                    </span>

                    <span>
                      H{" "}
                      <b className="font-medium text-green-400">
                        {formatPrice(
                          stock?.high
                        )}
                      </b>
                    </span>

                    <span>
                      L{" "}
                      <b className="font-medium text-red-400">
                        {formatPrice(
                          stock?.low
                        )}
                      </b>
                    </span>

                    <span>
                      C{" "}
                      <b className="font-medium text-gray-400">
                        {formatPrice(
                          currentPrice
                        )}
                      </b>
                    </span>

                    <span>
                      Vol{" "}
                      <b className="font-medium text-gray-400">
                        {formatVolume(
                          stock?.volume
                        )}
                      </b>
                    </span>

                  </div>

                </div>


                <div className="flex flex-wrap gap-1 rounded-xl bg-[#090e16] p-1">

                  {RANGE_OPTIONS.map(
                    (
                      range
                    ) => (

                      <button
                        key={
                          range.key
                        }
                        onClick={() =>
                          setSelectedRange(
                            range.key
                          )
                        }
                        className={`rounded-lg px-2.5 py-2 text-[11px] font-medium transition ${
                          selectedRange ===
                          range.key
                            ? "bg-indigo-500 text-white"
                            : "text-gray-500 hover:bg-white/5 hover:text-gray-300"
                        }`}
                      >
                        {range.label}
                      </button>

                    )
                  )}

                </div>

              </div>

            </div>


            <div className="px-1 pb-1">

              {stockLoading &&
              !stock ? (
                <LoadingBox
                  text="Loading market chart..."
                />
              ) : (
                <CandlestickStockChart
                  candles={
                    candleData
                  }
                  currentPrice={
                    stock?.price
                  }
                  previousClose={
                    stock?.previous_close
                  }
                  rangeLabel={
                    RANGE_OPTIONS.find(
                      (
                        item
                      ) =>
                        item.key ===
                        selectedRange
                    )?.label ||
                    "3M"
                  }
                  loading={
                    candleLoading
                  }
                  error={
                    candleError
                  }
                  theme={
                    theme
                  }
                />
              )}

            </div>

          </div>


          {/* AI COLUMN */}

          <div className="grid gap-3">

            {/* NEXT DAY */}

            <div className="rounded-xl border border-[#1b2738] bg-[#0f141d] p-4">

              <div className="flex items-center justify-between gap-3">

                <div>

                  <div className="flex items-center gap-2">

                    <div className="rounded-lg bg-purple-500/10 p-2">
                      <Bot
                        size={16}
                        className="text-purple-400"
                      />
                    </div>

                    <h2 className="text-sm font-semibold text-white">
                      AI Prediction (Next Day)
                    </h2>

                  </div>

                  <p className="mt-1 pl-10 text-[10px] text-gray-600">
                    BiLSTM
                  </p>

                </div>

                <TrendBadge
                  signal={
                    forecastSignal
                  }
                />

              </div>


              <div className="mt-4 rounded-lg border border-green-500/15 bg-[linear-gradient(100deg,rgba(16,185,129,0.17),rgba(16,185,129,0.045))] p-4">

                <div className="flex items-end justify-between gap-4">

                  <div>

                    <p className="text-[10px] text-gray-500">
                      Potential Price
                    </p>

                    <p className="mt-1 text-2xl font-bold text-white">
                      {predictionLoading
                        ? "..."
                        : formatPrice(
                            prediction?.predicted_price
                          )}
                    </p>

                  </div>


                  <div className="text-right">

                    <p className="text-[10px] text-gray-500">
                      Expected Move
                    </p>

                    <p
                      className={`mt-1 text-xl font-bold ${
                        forecastMove >= 0
                          ? "text-green-400"
                          : "text-red-400"
                      }`}
                    >
                      {predictionLoading
                        ? "..."
                        : formatPercent(
                            forecastMove,
                            4
                          )}
                    </p>

                  </div>

                </div>

              </div>


              <div className="mt-3 flex items-center justify-between rounded-xl border border-white/5 bg-black/10 px-3 py-3">

                <div>
                  <p className="text-[10px] text-gray-600">
                    Model
                  </p>
                  <p className="mt-1 text-xs font-medium text-white">
                    {prediction?.model ||
                      "BiLSTM"}
                  </p>
                </div>

                <button
                  onClick={() =>
                    setActivePage(
                      "AI Prediction"
                    )
                  }
                  className="text-[11px] font-medium text-purple-400"
                >
                  View Details
                </button>

              </div>

            </div>


            {/* V9 */}

            <div className="rounded-xl border border-[#1b2738] bg-[#0f141d] p-4">

              <div className="flex items-center justify-between gap-3">

                <div>
                  <h2 className="text-sm font-semibold text-white">
                    V9 Relative Strength vs NIFTY 50
                  </h2>
                  <p className="mt-1 text-[10px] text-gray-600">
                    5 trading days
                  </p>
                </div>

                <RelativeStrengthBadge
                  signal={
                    relativeSignal
                  }
                />

              </div>


              <div className="mt-4 rounded-lg border border-green-500/10 bg-[linear-gradient(100deg,rgba(16,185,129,0.09),rgba(16,185,129,0.02))] p-4">

                <div className="flex items-end justify-between gap-4">

                  <div>
                    <p className="text-[10px] text-gray-600">
                      Relative Signal
                    </p>
                    <p
                      className={`mt-1 text-xl font-bold ${
                        relativeSignal ===
                        "OUTPERFORM"
                          ? "text-green-400"
                          : relativeSignal ===
                            "UNDERPERFORM"
                          ? "text-red-400"
                          : "text-yellow-400"
                      }`}
                    >
                      {relativeSignal}
                    </p>
                  </div>

                  <div className="text-right">
                    <p className="text-[10px] text-gray-600">
                      Top raw score
                    </p>
                    <p className="mt-1 text-xl font-bold text-white">
                      {relativeLoading
                        ? "..."
                        : formatProbabilityScore(
                            relativePrediction?.top_probability
                          )}
                    </p>
                  </div>

                </div>


                <div className="mt-4">

                  <div className="mb-1 flex justify-between text-[9px] text-gray-600">
                    <span>
                      Model score
                    </span>
                    <span>
                      Not calibrated confidence
                    </span>
                  </div>

                  <div className="h-1.5 overflow-hidden rounded-full bg-white/5">
                    <div
                      className={`h-full rounded-full ${
                        relativeSignal ===
                        "OUTPERFORM"
                          ? "bg-green-400"
                          : relativeSignal ===
                            "UNDERPERFORM"
                          ? "bg-red-400"
                          : "bg-yellow-400"
                      }`}
                      style={{
                        width:
                          `${Math.max(
                            0,
                            Math.min(
                              100,
                              Number(
                                relativePrediction?.top_probability ||
                                0
                              ) *
                              100
                            )
                          )}%`,
                      }}
                    />
                  </div>

                </div>

              </div>

            </div>

          </div>

        </section>


        {/* =================================================
            MID ROW
            LEFT HALF: STOCK PERFORMANCE + INDICATORS
            RIGHT: WHY THIS PREDICTION
        ================================================= */}

        <section className="grid gap-3 xl:grid-cols-[minmax(0,1.62fr)_minmax(340px,0.78fr)]">

          <div className="grid gap-3 md:grid-cols-2">

            {/* PERFORMANCE */}

            <div className="rounded-xl border border-[#1b2738] bg-[#0f141d] p-4">

              <div className="flex items-center justify-between">

                <div>
                  <h3 className="text-sm font-semibold text-white">
                    Stock Performance
                  </h3>
                  <p className="mt-1 text-[10px] text-gray-600">
                    Multi-period return
                  </p>
                </div>

                <LineChartIcon
                  size={16}
                  className="text-blue-400"
                />

              </div>


              <div className="mt-4 grid grid-cols-5 gap-2">

                {performanceCards.map(
                  (
                    label
                  ) => {

                    const value =
                      performanceData[
                        label
                      ];

                    const positive =
                      Number(
                        value
                      ) >= 0;

                    return (
                      <div
                        key={
                          label
                        }
                        className="rounded-xl border border-white/5 bg-[#0b1018] p-2.5 text-center"
                      >
                        <p className="text-[10px] text-gray-500">
                          {label}
                        </p>

                        <p
                          className={`mt-2 text-xs font-semibold ${
                            value === null ||
                            value === undefined
                              ? "text-gray-600"
                              : positive
                              ? "text-green-400"
                              : "text-red-400"
                          }`}
                        >
                          {performanceLoading &&
                          value === undefined
                            ? "..."
                            : formatPercent(
                                value
                              )}
                        </p>
                      </div>
                    );
                  }
                )}

              </div>

            </div>


            {/* TECHNICAL */}

            <div className="rounded-xl border border-[#1b2738] bg-[#0f141d] p-4">

              <div className="flex items-center justify-between">

                <div>
                  <h3 className="text-sm font-semibold text-white">
                    Technical Indicators
                  </h3>
                  <p className="mt-1 text-[10px] text-gray-600">
                    Momentum / trend
                  </p>
                </div>

                <Activity
                  size={16}
                  className="text-blue-400"
                />

              </div>


              <div className="mt-4 grid grid-cols-2 gap-x-5 gap-y-4">

                {indicatorItems.map(
                  (
                    item
                  ) => (

                    <div
                      key={
                        item.label
                      }
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-[10px] text-gray-500">
                          {item.label}
                        </span>
                        <span
                          className={`text-[10px] font-medium ${
                            item.positive
                              ? "text-green-400"
                              : "text-gray-500"
                          }`}
                        >
                          {item.status}
                        </span>
                      </div>

                      <p className="mt-1 text-sm font-semibold text-white">
                        {item.value}
                      </p>
                    </div>

                  )
                )}

              </div>

            </div>

          </div>


          {/* WHY THIS PREDICTION */}

          <div className="rounded-xl border border-[#1b2738] bg-[#0f141d] p-4">

            <div className="flex items-center justify-between gap-3">

              <div>
                <h3 className="text-sm font-semibold text-white">
                  Why This Prediction?
                </h3>
                <p className="mt-1 text-[10px] text-gray-600">
                  Technical + relative context
                </p>
              </div>

              <RelativeStrengthBadge
                signal={
                  relativeSignal
                }
              />

            </div>


            <div className="mt-4 space-y-2.5">

              {whyItems
                .slice(
                  0,
                  5
                )
                .map(
                  (
                    item,
                    index
                  ) => (

                    <div
                      key={
                        `${index}-${item.text}`
                      }
                      className="flex items-start gap-2.5"
                    >
                      {item.positive ? (
                        <TrendingUp
                          size={14}
                          className="mt-0.5 shrink-0 text-green-400"
                        />
                      ) : (
                        <TrendingDown
                          size={14}
                          className="mt-0.5 shrink-0 text-red-400"
                        />
                      )}

                      <p className="text-[11px] leading-5 text-gray-400">
                        {item.text}
                      </p>
                    </div>

                  )
                )}

            </div>


            <div className="mt-4 border-t border-white/5 pt-3">
              <p className="text-[10px] leading-4 text-gray-600">
                Explanation summarizes visible model inputs and market context; it is not a causal claim.
              </p>
            </div>

          </div>

        </section>


        {/* =================================================
            BOTTOM ROW
            LEFT: TOP GAINERS / LOSERS
            RIGHT: MODEL MONITOR
        ================================================= */}

        <section className="grid gap-3 xl:grid-cols-[minmax(0,1.62fr)_minmax(340px,0.78fr)]">

          {/* MOVERS */}

          <div className="rounded-xl border border-[#1b2738] bg-[#0f141d] p-4">

            <div className="grid gap-6 md:grid-cols-2">

              <div>

                <div className="mb-3 flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-white">
                    Top Gainers
                  </h3>
                  <TrendingUp
                    size={15}
                    className="text-green-400"
                  />
                </div>


                <div className="space-y-1">

                  {gainers.map(
                    (
                      item,
                      index
                    ) => (

                      <button
                        key={
                          `gainer-${item.symbol}`
                        }
                        onClick={() =>
                          selectStock(
                            item.symbol
                          )
                        }
                        className="grid w-full grid-cols-[24px_1fr_auto_auto] items-center gap-3 rounded-lg px-2 py-2 text-left hover:bg-white/[0.03]"
                      >
                        <span className="text-[10px] text-gray-700">
                          {index + 1}
                        </span>

                        <span className="truncate text-xs font-medium text-gray-300">
                          {item.short}
                        </span>

                        <span className="text-[11px] text-gray-500">
                          {formatPrice(
                            item.price
                          )}
                        </span>

                        <span className="min-w-[56px] text-right text-[11px] font-semibold text-green-400">
                          {formatPercent(
                            item.change_percent
                          )}
                        </span>
                      </button>

                    )
                  )}

                  {trackedStocksLoading &&
                  gainers.length ===
                    0 && (
                    <p className="py-4 text-xs text-gray-600">
                      Loading movers...
                    </p>
                  )}

                </div>

              </div>


              <div>

                <div className="mb-3 flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-white">
                    Top Losers
                  </h3>
                  <TrendingDown
                    size={15}
                    className="text-red-400"
                  />
                </div>


                <div className="space-y-1">

                  {losers.map(
                    (
                      item,
                      index
                    ) => (

                      <button
                        key={
                          `loser-${item.symbol}`
                        }
                        onClick={() =>
                          selectStock(
                            item.symbol
                          )
                        }
                        className="grid w-full grid-cols-[24px_1fr_auto_auto] items-center gap-3 rounded-lg px-2 py-2 text-left hover:bg-white/[0.03]"
                      >
                        <span className="text-[10px] text-gray-700">
                          {index + 1}
                        </span>

                        <span className="truncate text-xs font-medium text-gray-300">
                          {item.short}
                        </span>

                        <span className="text-[11px] text-gray-500">
                          {formatPrice(
                            item.price
                          )}
                        </span>

                        <span className="min-w-[56px] text-right text-[11px] font-semibold text-red-400">
                          {formatPercent(
                            item.change_percent
                          )}
                        </span>
                      </button>

                    )
                  )}

                </div>

              </div>

            </div>

          </div>


          {/* PREDICTION HISTORY */}

          <div className="rounded-xl border border-[#1b2738] bg-[#0f141d] p-4">

            <div className="flex items-center justify-between">

              <div>
                <h3 className="text-sm font-semibold text-white">
                  Prediction History
                </h3>
                <p className="mt-1 text-[10px] text-gray-600">
                  Saved live V9 outputs
                </p>
              </div>

              <button
                onClick={() =>
                  setActivePage(
                    "AI Prediction"
                  )
                }
                className="text-[10px] font-medium text-indigo-400"
              >
                View All
              </button>

            </div>


            <div className="mt-3 overflow-hidden rounded-xl border border-white/5">

              <div className="grid grid-cols-[66px_1fr_auto_auto] gap-2 border-b border-white/5 bg-white/[0.02] px-3 py-2 text-[8px] uppercase tracking-wider text-gray-600">
                <span>Date</span>
                <span>Stock</span>
                <span>Signal</span>
                <span>Outcome</span>
              </div>


              {predictionHistory
                .slice(
                  0,
                  5
                )
                .map(
                  (
                    item
                  ) => (

                    <div
                      key={
                        item.key
                      }
                      className="grid grid-cols-[66px_1fr_auto_auto] items-center gap-2 border-b border-white/[0.035] px-3 py-2 last:border-0"
                    >

                      <span className="text-[9px] text-gray-600">
                        {item.date}
                      </span>

                      <span className="truncate text-[10px] font-medium text-gray-300">
                        {item.stock}
                      </span>

                      <span
                        className={`text-[9px] font-semibold ${
                          item.signal ===
                          "OUTPERFORM"
                            ? "text-green-400"
                            : item.signal ===
                              "UNDERPERFORM"
                            ? "text-red-400"
                            : "text-yellow-400"
                        }`}
                      >
                        {item.signal}
                      </span>

                      <span className="text-[9px] text-gray-600">
                        {item.outcome}
                      </span>

                    </div>

                  )
                )}


              {predictionHistory.length ===
                0 && (

                <div className="px-3 py-6 text-center">

                  <p className="text-[10px] text-gray-600">
                    History will appear as live V9 predictions are generated.
                  </p>

                </div>

              )}

            </div>

          </div>

        </section>


        <div className="rounded-xl border border-yellow-500/10 bg-yellow-500/[0.03] px-4 py-3">

          <p className="text-[10px] leading-4 text-gray-600">
            StockVision is an educational AI market-intelligence project. Market data may be delayed, and model outputs are not investment advice.
          </p>

        </div>

      </div>
    );
  }


  // =======================================================
  // WATCHLIST
  // =======================================================

  function WatchlistPage() {

    const [
      watchlistSearch,
      setWatchlistSearch,
    ] = useState(
      ""
    );


    const watchlistSearchResults =
      useMemo(
        () => {

          const query =
            watchlistSearch
              .trim()
              .toLowerCase();


          if (!query) {

            return [];
          }


          return allStocks
            .filter(
              (
                item
              ) =>

                item.name
                  .toLowerCase()
                  .includes(
                    query
                  )

                ||

                item.short
                  .toLowerCase()
                  .includes(
                    query
                  )

                ||

                item.symbol
                  .toLowerCase()
                  .includes(
                    query
                  )
            )
            .slice(
              0,
              10
            );

        },
        [
          watchlistSearch,
          allStocks,
        ]
      );


    return (
      <div>

        <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-end">

          <div>

            <h1 className="text-2xl font-bold text-white">
              My Watchlist
            </h1>


            <p className="mt-1 text-sm text-gray-500">
              Add any NSE stock and keep your favourites in one place.
            </p>

          </div>


          <div className="rounded-xl border border-white/5 bg-[#0f141d] px-4 py-2.5">

            <p className="text-[9px] uppercase tracking-[0.14em] text-gray-600">
              Saved Stocks
            </p>

            <p className="mt-1 text-lg font-bold text-yellow-400">
              {watchlistItems.length}
            </p>

          </div>

        </div>


        {/* ADD STOCK SEARCH */}

        <div className="relative mt-6">

          <Search
            size={16}
            className="absolute left-4 top-[18px] text-gray-600"
          />


          <input
            value={
              watchlistSearch
            }
            onChange={
              (
                event
              ) =>
                setWatchlistSearch(
                  event.target.value
                )
            }
            placeholder="Search any NSE stock to add to your watchlist..."
            className="w-full rounded-xl border border-white/5 bg-[#0f141d] py-3.5 pl-11 pr-4 text-sm text-white outline-none transition focus:border-yellow-500/30"
          />


          {watchlistSearch.trim() &&
          watchlistSearchResults.length >
            0 && (

            <div className="absolute left-0 right-0 top-[58px] z-30 max-h-[390px] overflow-y-auto rounded-xl border border-white/10 bg-[#10151f] p-1.5 shadow-2xl">

              {watchlistSearchResults.map(
                (
                  item
                ) => {

                  const added =
                    isWatchlisted(
                      item.symbol
                    );


                  return (
                    <div
                      key={
                        `watchlist-search-${item.symbol}`
                      }
                      className="flex items-center justify-between gap-3 rounded-lg px-3 py-2.5 transition hover:bg-white/[0.04]"
                    >

                      <button
                        onClick={() => {

                          selectStock(
                            item.symbol
                          );

                          setWatchlistSearch(
                            ""
                          );

                          setActivePage(
                            "Dashboard"
                          );

                        }}
                        className="min-w-0 flex-1 text-left"
                      >

                        <p className="truncate text-sm font-medium text-white">
                          {item.name}
                        </p>

                        <p className="mt-0.5 text-[10px] text-gray-500">
                          {item.symbol}
                        </p>

                      </button>


                      <button
                        onClick={() =>
                          toggleWatchlist(
                            item
                          )
                        }
                        className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg transition ${
                          added
                            ? "bg-yellow-500/10 text-yellow-400"
                            : "bg-white/[0.03] text-gray-500 hover:text-yellow-400"
                        }`}
                        title={
                          added
                            ? "Remove from watchlist"
                            : "Add to watchlist"
                        }
                      >

                        <Star
                          size={16}
                          fill={
                            added
                              ? "currentColor"
                              : "none"
                          }
                        />

                      </button>

                    </div>
                  );
                }
              )}

            </div>

          )}

        </div>


        {/* WATCHLIST CARDS */}

        {watchlistItems.length ===
          0 ? (

          <div className="mt-6 rounded-2xl border border-dashed border-white/10 bg-[#0f141d] px-6 py-14 text-center">

            <Star
              size={30}
              className="mx-auto text-gray-700"
            />

            <p className="mt-4 text-sm font-semibold text-gray-300">
              Your watchlist is empty
            </p>

            <p className="mt-1 text-xs text-gray-600">
              Search above and press the star to add any NSE stock.
            </p>

          </div>

        ) : (

          <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-3">

            {watchlistItems.map(
              (
                item
              ) => {

                const live =
                  watchlistLiveData[
                    item.symbol
                  ];


                const move =
                  Number(
                    live?.change_percent
                  );


                const positive =
                  Number.isFinite(
                    move
                  )
                    ? move >= 0
                    : null;


                return (
                  <div
                    key={
                      item.symbol
                    }
                    className="rounded-2xl border border-[#1b2738] bg-[#0f141d] p-5 transition hover:border-blue-500/25"
                  >

                    <div className="flex items-start justify-between gap-3">

                      <button
                        onClick={() => {

                          selectStock(
                            item.symbol
                          );

                          setActivePage(
                            "Dashboard"
                          );

                        }}
                        className="min-w-0 text-left"
                      >

                        <p className="truncate text-base font-semibold text-white">
                          {item.short}
                        </p>

                        <p className="mt-1 truncate text-xs text-gray-500">
                          {item.name}
                        </p>

                      </button>


                      <button
                        onClick={() =>
                          toggleWatchlist(
                            item.symbol
                          )
                        }
                        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-yellow-500/10 text-yellow-400 transition hover:bg-red-500/10 hover:text-red-400"
                        title="Remove from watchlist"
                      >

                        <Star
                          size={17}
                          fill="currentColor"
                        />

                      </button>

                    </div>


                    <button
                      onClick={() => {

                        selectStock(
                          item.symbol
                        );

                        setActivePage(
                          "Dashboard"
                        );

                      }}
                      className="mt-5 w-full text-left"
                    >

                      <div className="flex items-end justify-between gap-4">

                        <div>

                          <p className="text-[10px] uppercase tracking-wider text-gray-600">
                            Live / Latest Price
                          </p>

                          <p className="mt-1 text-xl font-bold text-white">

                            {watchlistLoading &&
                            !live
                              ? "..."
                              : live
                              ? formatPrice(
                                  live.price
                                )
                              : "--"}

                          </p>

                        </div>


                        <div className="text-right">

                          <p className="text-[10px] uppercase tracking-wider text-gray-600">
                            Today
                          </p>

                          <p
                            className={`mt-1 text-sm font-semibold ${
                              positive ===
                              null
                                ? "text-gray-500"
                                : positive
                                ? "text-green-400"
                                : "text-red-400"
                            }`}
                          >

                            {Number.isFinite(
                              move
                            )
                              ? formatPercent(
                                  move
                                )
                              : "--"}

                          </p>

                        </div>

                      </div>


                      <div className="mt-4 flex items-center justify-between border-t border-white/5 pt-3">

                        <span className="text-[10px] text-gray-600">
                          {item.symbol}
                        </span>

                        <span className="text-[10px] font-medium text-blue-400">
                          Open Dashboard →
                        </span>

                      </div>

                    </button>

                  </div>
                );
              }
            )}

          </div>

        )}

      </div>
    );
  }


  // =======================================================
  // MARKETS
  // =======================================================

  function MarketsPage() {

    const [
      marketsSearch,
      setMarketsSearch,
    ] = useState(
      ""
    );


    const [
      visibleStockCount,
      setVisibleStockCount,
    ] = useState(
      100
    );


    const filteredMarketStocks =
      useMemo(
        () => {

          const query =
            marketsSearch
              .trim()
              .toLowerCase();


          if (!query) {

            return allStocks;
          }


          return allStocks.filter(
            (
              item
            ) =>

              item.name
                .toLowerCase()
                .includes(
                  query
                )

              ||

              item.short
                .toLowerCase()
                .includes(
                  query
                )

              ||

              item.symbol
                .toLowerCase()
                .includes(
                  query
                )
          );

        },
        [
          marketsSearch,
          allStocks,
        ]
      );


    const visibleMarketStocks =
      filteredMarketStocks.slice(
        0,
        visibleStockCount
      );


    return (
      <div>

        <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-end">

          <div>

            <h1 className="text-2xl font-bold text-white">
              NSE Stocks
            </h1>


            <p className="mt-1 text-sm text-gray-500">

              {stockUniverseLoading
                ? "Loading current NSE stock universe..."
                : `${allStocks.length.toLocaleString()} NSE securities available in StockVision.`}

            </p>

          </div>


          <div className="rounded-xl border border-white/5 bg-[#0f141d] px-3 py-2">

            <p className="text-[9px] uppercase tracking-wider text-gray-600">
              Stock Universe
            </p>

            <p className="mt-1 text-xs font-medium text-blue-400">
              {stockUniverseSource}
            </p>

          </div>

        </div>


        <div className="relative mt-5">

          <Search
            size={16}
            className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-600"
          />


          <input
            value={
              marketsSearch
            }
            onChange={
              (
                event
              ) => {

                setMarketsSearch(
                  event.target.value
                );

                setVisibleStockCount(
                  100
                );
              }
            }
            placeholder="Search any NSE stock by company name or symbol..."
            className="w-full rounded-xl border border-white/5 bg-[#0f141d] py-3 pl-11 pr-4 text-sm text-white outline-none transition focus:border-blue-500/30"
          />

        </div>


        <div className="mt-3 flex items-center justify-between">

          <p className="text-xs text-gray-600">

            {filteredMarketStocks.length.toLocaleString()}
            {" "}
            matching stocks

          </p>


          <p className="text-[10px] text-gray-700">
            Showing {Math.min(
              visibleStockCount,
              filteredMarketStocks.length
            ).toLocaleString()}
          </p>

        </div>


        <div className="mt-3 overflow-hidden rounded-xl border border-[#1b2738] bg-[#0f141d]">

          {visibleMarketStocks.map(
            (
              item
            ) => (

              <div
                key={
                  item.symbol
                }
                className="flex items-center justify-between gap-4 border-b border-white/5 px-5 py-3.5 last:border-0"
              >

                <div className="min-w-0">

                  <p className="truncate text-sm font-medium text-white">
                    {item.name}
                  </p>


                  <div className="mt-1 flex flex-wrap items-center gap-2">

                    <p className="text-xs text-gray-500">
                      {item.symbol}
                    </p>


                    {item.series && (

                      <span className="rounded bg-white/[0.04] px-1.5 py-0.5 text-[8px] font-medium text-gray-600">
                        {item.series}
                      </span>

                    )}

                  </div>

                </div>


                <div className="flex shrink-0 items-center gap-2">

                  <button
                    onClick={() =>
                      toggleWatchlist(
                        item
                      )
                    }
                    title={
                      isWatchlisted(
                        item.symbol
                      )
                        ? "Remove from watchlist"
                        : "Add to watchlist"
                    }
                    className={`flex h-9 w-9 items-center justify-center rounded-lg transition ${
                      isWatchlisted(
                        item.symbol
                      )
                        ? "bg-yellow-500/10 text-yellow-400"
                        : "bg-white/[0.03] text-gray-600 hover:text-yellow-400"
                    }`}
                  >

                    <Star
                      size={15}
                      fill={
                        isWatchlisted(
                          item.symbol
                        )
                          ? "currentColor"
                          : "none"
                      }
                    />

                  </button>


                  <button
                    onClick={() => {

                      selectStock(
                        item.symbol
                      );

                      setActivePage(
                        "Dashboard"
                      );

                    }}
                    className="rounded-lg bg-blue-500/10 px-3 py-2 text-xs font-medium text-blue-400 transition hover:bg-blue-500/15"
                  >
                    View
                  </button>

                </div>

              </div>

            )
          )}


          {visibleMarketStocks.length ===
            0 && (

            <div className="px-5 py-12 text-center">

              <Search
                size={24}
                className="mx-auto text-gray-700"
              />

              <p className="mt-3 text-sm text-gray-400">
                No matching NSE stock found.
              </p>

            </div>

          )}

        </div>


        {visibleStockCount <
          filteredMarketStocks.length && (

          <div className="mt-4 flex justify-center">

            <button
              onClick={() =>
                setVisibleStockCount(
                  (
                    current
                  ) =>
                    current +
                    100
                )
              }
              className="rounded-xl border border-white/5 bg-[#0f141d] px-5 py-2.5 text-xs font-medium text-gray-300 transition hover:border-blue-500/20 hover:text-blue-400"
            >
              Load 100 More
            </button>

          </div>

        )}

      </div>
    );
  }


  // =======================================================
  // AI PREDICTION PAGE
  // =======================================================

  function AIPredictionPage() {

    const [
      predictionValidation,
      setPredictionValidation,
    ] = useState(
      null
    );


    const [
      predictionValidationLoading,
      setPredictionValidationLoading,
    ] = useState(
      false
    );


    const [
      predictionValidationError,
      setPredictionValidationError,
    ] = useState(
      ""
    );


    const [
      historySymbol,
      setHistorySymbol,
    ] = useState(
      selectedSymbol
    );


    const [
      historySearch,
      setHistorySearch,
    ] = useState(
      ""
    );


    const [
      historySearchOpen,
      setHistorySearchOpen,
    ] = useState(
      false
    );


    const [
      historyTrackingMessage,
      setHistoryTrackingMessage,
    ] = useState(
      ""
    );


    const [
      historyOverview,
      setHistoryOverview,
    ] = useState(
      null
    );


    const [
      historyOverviewLoading,
      setHistoryOverviewLoading,
    ] = useState(
      false
    );


    const [
      historyOverviewError,
      setHistoryOverviewError,
    ] = useState(
      ""
    );


    const [
      fullNseCapture,
      setFullNseCapture,
    ] = useState(
      null
    );


    const [
      fullNseCaptureLoading,
      setFullNseCaptureLoading,
    ] = useState(
      false
    );


    const [
      historyOverviewSearch,
      setHistoryOverviewSearch,
    ] = useState(
      ""
    );


    const [
      historyOverviewStatus,
      setHistoryOverviewStatus,
    ] = useState(
      "ALL"
    );


    const [
      historyOverviewPage,
      setHistoryOverviewPage,
    ] = useState(
      1
    );


    const HISTORY_OVERVIEW_PAGE_SIZE =
      25;


    const historySearchRef =
      useRef(
        null
      );


    const historyDetailRef =
      useRef(
        null
      );


    const historyRequestRef =
      useRef(
        0
      );


    const [
      historyDateRange,
      setHistoryDateRange,
    ] = useState(
      "ALL"
    );


    const [
      historyChartExpanded,
      setHistoryChartExpanded,
    ] = useState(
      false
    );


    const historyStockInfo =
      allStocks.find(
        (
          item
        ) =>
          item.symbol ===
          historySymbol
      ) ||
      {
        symbol:
          historySymbol,
        short:
          historySymbol.replace(
            ".NS",
            ""
          ),
        name:
          historySymbol.replace(
            ".NS",
            ""
          ),
      };


    const filteredPredictionHistory =
      useMemo(
        () => {

          const rows =
            Array.isArray(
              predictionValidation?.history
            )
              ? predictionValidation.history
              : [];


          if (
            historyDateRange ===
            "ALL"
          ) {

            return rows;
          }


          const dayMap = {
            "7D":
              7,
            "30D":
              30,
            "3M":
              90,
            "6M":
              180,
            "1Y":
              365,
          };


          const days =
            dayMap[
              historyDateRange
            ] ||
            null;


          if (
            !days
          ) {

            return rows;
          }


          const validDates =
            rows
              .map(
                (
                  row
                ) =>
                  row.base_date ||
                  row.target_date
              )
              .filter(
                Boolean
              )
              .map(
                (
                  value
                ) =>
                  new Date(
                    `${value}T00:00:00`
                  )
              )
              .filter(
                (
                  date
                ) =>
                  !Number.isNaN(
                    date.getTime()
                  )
              );


          if (
            validDates.length ===
            0
          ) {

            return rows;
          }


          const latestDate =
            new Date(
              Math.max(
                ...validDates.map(
                  (
                    date
                  ) =>
                    date.getTime()
                )
              )
            );


          const cutoff =
            new Date(
              latestDate
            );


          cutoff.setDate(
            cutoff.getDate() -
            (
              days -
              1
            )
          );


          return rows.filter(
            (
              row
            ) => {

              const rawDate =
                row.base_date ||
                row.target_date;


              if (
                !rawDate
              ) {

                return false;
              }


              const parsed =
                new Date(
                  `${rawDate}T00:00:00`
                );


              if (
                Number.isNaN(
                  parsed.getTime()
                )
              ) {

                return false;
              }


              return (
                parsed >=
                cutoff
                &&
                parsed <=
                latestDate
              );
            }
          );

        },
        [
          predictionValidation
            ?.history,
          historyDateRange,
        ]
      );


    useEffect(
      () => {

        if (
          !historyChartExpanded
        ) {

          return;
        }


        const previousOverflow =
          document.body.style
            .overflow;


        document.body.style.overflow =
          "hidden";


        const handleKeyDown =
          (
            event
          ) => {

            if (
              event.key ===
              "Escape"
            ) {

              setHistoryChartExpanded(
                false
              );
            }
          };


        window.addEventListener(
          "keydown",
          handleKeyDown
        );


        return () => {

          document.body.style.overflow =
            previousOverflow;

          window.removeEventListener(
            "keydown",
            handleKeyDown
          );
        };

      },
      [
        historyChartExpanded,
      ]
    );


    const historyChartData =
      useMemo(
        () => {

          const toFiniteOrNull =
            (
              value
            ) => {

              if (
                value === null ||
                value === undefined ||
                value === ""
              ) {

                return null;
              }


              const number =
                Number(
                  value
                );


              return Number.isFinite(
                number
              )
                ? number
                : null;
            };


          return filteredPredictionHistory
            .map(
              (
                row
              ) => {

                const predicted =
                  toFiniteOrNull(
                    row.forecast_point_price ??
                    row.experimental_x2_point_price ??
                    row.predicted_price
                  );


                const actual =
                  toFiniteOrNull(
                    row.actual_close
                  );


                const rangeLow =
                  toFiniteOrNull(
                    row.expected_range_lower
                  );


                const rangeHigh =
                  toFiniteOrNull(
                    row.expected_range_upper
                  );


                const baseClose =
                  toFiniteOrNull(
                    row.current_close
                  );


                const rawDate =
                  row.target_date ||
                  row.base_date ||
                  "";


                let label =
                  rawDate;


                if (
                  rawDate
                ) {

                  const parsed =
                    new Date(
                      `${rawDate}T00:00:00`
                    );


                  if (
                    !Number.isNaN(
                      parsed.getTime()
                    )
                  ) {

                    label =
                      parsed.toLocaleDateString(
                        [],
                        {
                          day:
                            "2-digit",
                          month:
                            "short",
                        }
                      );
                  }
                }


                const difference =
                  actual !==
                    null &&
                  predicted !==
                    null
                    ? (
                        actual -
                        predicted
                      )
                    : null;


                const finalValue =
                  actual !==
                  null
                    ? actual
                    : predicted;


                return {
                  date:
                    label ||
                    "--",
                  baseClose,
                  predictedValue:
                    predicted,
                  actualValue:
                    actual,
                  difference,
                  finalValue,
                  rangeLow,
                  rangeHigh,
                };
              }
            )
            .reverse();

        },
        [
          filteredPredictionHistory,
        ]
      );


    const historyPerformance =
      useMemo(
        () => {

          const toFiniteOrNull =
            (
              value
            ) => {

              if (
                value === null ||
                value === undefined ||
                value === ""
              ) {

                return null;
              }


              const number =
                Number(
                  value
                );


              return Number.isFinite(
                number
              )
                ? number
                : null;
            };


          const calculate =
            (
              rows
            ) => {

              const resolved =
                rows.filter(
                  (
                    row
                  ) =>
                    row.status ===
                    "RESOLVED"
                );


              const priceRows =
                resolved
                  .map(
                    (
                      row
                    ) => {

                      const predicted =
                        toFiniteOrNull(
                          row.forecast_point_price ??
                          row.experimental_x2_point_price ??
                          row.predicted_price
                        );


                      const actual =
                        toFiniteOrNull(
                          row.actual_close
                        );


                      const base =
                        toFiniteOrNull(
                          row.current_close ??
                          row.base_close
                        );


                      if (
                        predicted === null ||
                        actual === null
                      ) {

                        return null;
                      }


                      const absError =
                        Math.abs(
                          actual -
                          predicted
                        );


                      const errorPercent =
                        actual !==
                        0
                          ? (
                              absError /
                              Math.abs(
                                actual
                              )
                            ) *
                            100
                          : null;


                      const baselineError =
                        base !==
                        null
                          ? Math.abs(
                              actual -
                              base
                            )
                          : null;


                      return {
                        row,
                        predicted,
                        actual,
                        base,
                        absError,
                        errorPercent,
                        baselineError,
                      };
                    }
                  )
                  .filter(
                    Boolean
                  );


              const average =
                (
                  values
                ) => {

                  const clean =
                    values.filter(
                      (
                        value
                      ) =>
                        Number.isFinite(
                          value
                        )
                    );


                  if (
                    clean.length ===
                    0
                  ) {

                    return null;
                  }


                  return (
                    clean.reduce(
                      (
                        sum,
                        value
                      ) =>
                        sum +
                        value,
                      0
                    ) /
                    clean.length
                  );
                };


              const mae =
                average(
                  priceRows.map(
                    (
                      item
                    ) =>
                      item.absError
                  )
                );


              const mape =
                average(
                  priceRows.map(
                    (
                      item
                    ) =>
                      item.errorPercent
                  )
                );


              const baselineMae =
                average(
                  priceRows.map(
                    (
                      item
                    ) =>
                      item.baselineError
                  )
                );


              const baselineImprovement =
                mae !==
                  null &&
                baselineMae !==
                  null &&
                baselineMae >
                  0
                  ? (
                      (
                        baselineMae -
                        mae
                      ) /
                      baselineMae
                    ) *
                    100
                  : null;


              const directionRows =
                resolved.filter(
                  (
                    row
                  ) =>
                    row.direction_correct ===
                      true ||
                    row.direction_correct ===
                      false
                );


              const directionCorrect =
                directionRows.filter(
                  (
                    row
                  ) =>
                    row.direction_correct ===
                    true
                ).length;


              const directionAccuracy =
                directionRows.length >
                0
                  ? (
                      directionCorrect /
                      directionRows.length
                    ) *
                    100
                  : null;


              const rangeRows =
                resolved.filter(
                  (
                    row
                  ) =>
                    row.inside_expected_range ===
                      true ||
                    row.inside_expected_range ===
                      false
                );


              const rangeInside =
                rangeRows.filter(
                  (
                    row
                  ) =>
                    row.inside_expected_range ===
                    true
                ).length;


              const rangeCoverage =
                rangeRows.length >
                0
                  ? (
                      rangeInside /
                      rangeRows.length
                    ) *
                    100
                  : null;


              return {
                totalRows:
                  rows.length,
                resolvedRows:
                  resolved.length,
                priceSamples:
                  priceRows.length,
                directionSamples:
                  directionRows.length,
                rangeSamples:
                  rangeRows.length,
                directionAccuracy,
                mae,
                mape,
                rangeCoverage,
                baselineMae,
                baselineImprovement,
              };
            };


          const liveRows =
            filteredPredictionHistory.filter(
              (
                row
              ) =>
                row.history_source !==
                "BACKFILLED_MODEL_REPLAY"
            );


          const replayRows =
            filteredPredictionHistory.filter(
              (
                row
              ) =>
                row.history_source ===
                "BACKFILLED_MODEL_REPLAY"
            );


          return {
            live:
              calculate(
                liveRows
              ),
            replay:
              calculate(
                replayRows
              ),
            all:
              calculate(
                filteredPredictionHistory
              ),
          };

        },
        [
          filteredPredictionHistory,
        ]
      );


    const historySearchMatches =
      useMemo(
        () => {

          const clean =
            historySearch
              .trim()
              .toLowerCase();


          if (
            !clean
          ) {

            return allStocks.slice(
              0,
              8
            );
          }


          return allStocks
            .filter(
              (
                item
              ) =>
                item.short
                  .toLowerCase()
                  .includes(
                    clean
                  )
                ||
                item.name
                  .toLowerCase()
                  .includes(
                    clean
                  )
                ||
                item.symbol
                  .toLowerCase()
                  .includes(
                    clean
                  )
            )
            .slice(
              0,
              8
            );

        },
        [
          historySearch,
          allStocks,
        ]
      );


    async function fetchFullNseCaptureStatus() {

      try {

        const response =
          await fetch(
            `${API_URL}/prediction-full-universe/status`
          );


        const data =
          await response
            .json()
            .catch(
              () =>
                null
            );


        if (
          response.ok
          && data
        ) {

          setFullNseCapture(
            data
          );
        }

      } catch {

        // History page can continue even if status polling fails.
      }
    }


    async function startFullNseCapture() {

      setFullNseCaptureLoading(
        true
      );


      try {

        const response =
          await fetch(
            `${API_URL}/prediction-full-universe/start`,
            {
              method:
                "POST",
            }
          );


        const data =
          await response
            .json()
            .catch(
              () =>
                null
            );


        if (
          !response.ok
        ) {

          throw new Error(
            data?.detail ||
            "Unable to start full NSE capture."
          );
        }


        setFullNseCapture(
          data
        );


      } catch (
        error
      ) {

        setHistoryOverviewError(
          error.message ||
          "Unable to start full NSE capture."
        );


      } finally {

        setFullNseCaptureLoading(
          false
        );
      }
    }


    async function fetchHistoryOverview(
      pageOverride =
        historyOverviewPage,
      queryOverride =
        historyOverviewSearch,
      statusOverride =
        historyOverviewStatus
    ) {

      setHistoryOverviewLoading(
        true
      );

      setHistoryOverviewError(
        ""
      );


      try {

        const params =
          new URLSearchParams({
            query:
              queryOverride,
            page:
              String(
                pageOverride
              ),
            page_size:
              String(
                HISTORY_OVERVIEW_PAGE_SIZE
              ),
            status:
              statusOverride,
          });


        const response =
          await fetch(
            `${API_URL}/prediction-history-overview?${params.toString()}`
          );


        const data =
          await response
            .json()
            .catch(
              () =>
                null
            );


        if (
          !response.ok
        ) {

          throw new Error(
            data?.detail ||
            "Unable to load all-stock history."
          );
        }


        setHistoryOverview(
          data
        );


      } catch (
        error
      ) {

        setHistoryOverviewError(
          error.message ||
          "Unable to load all-stock history."
        );


      } finally {

        setHistoryOverviewLoading(
          false
        );
      }
    }


    async function fetchPredictionValidation(
      symbolOverride =
        historySymbol
    ) {

      const targetSymbol =
        typeof symbolOverride ===
        "string"
          ? symbolOverride
          : historySymbol;


      const requestId =
        historyRequestRef.current +
        1;


      historyRequestRef.current =
        requestId;


      try {

        setPredictionValidationLoading(
          true
        );

        setPredictionValidationError(
          ""
        );


        const response =
          await fetch(
            `${API_URL}/prediction-history/${encodeURIComponent(
              targetSymbol
            )}`
          );


        const data =
          await response
            .json()
            .catch(
              () =>
                null
            );


        if (
          !response.ok
        ) {

          throw new Error(
            data?.detail ||
            "Unable to validate predictions."
          );
        }


        if (
          historyRequestRef.current ===
          requestId
        ) {

          setPredictionValidation(
            data
          );
        }


      } catch (
        error
      ) {

        if (
          historyRequestRef.current ===
          requestId
        ) {

          setPredictionValidationError(
            error.message ||
            "Unable to validate predictions."
          );
        }


      } finally {

        if (
          historyRequestRef.current ===
          requestId
        ) {

          setPredictionValidationLoading(
            false
          );
        }
      }
    }


    useEffect(
      () => {

        setHistorySymbol(
          selectedSymbol
        );

        setHistorySearch(
          ""
        );

      },
      [
        selectedSymbol,
      ]
    );


    useEffect(
      () => {

        fetchPredictionValidation(
          historySymbol
        );

      },
      [
        historySymbol,
      ]
    );


    useEffect(
      () => {

        function handleHistorySearchOutside(
          event
        ) {

          if (
            historySearchRef.current
            &&
            !historySearchRef.current.contains(
              event.target
            )
          ) {

            setHistorySearchOpen(
              false
            );
          }
        }


        document.addEventListener(
          "mousedown",
          handleHistorySearchOutside
        );


        return () => {

          document.removeEventListener(
            "mousedown",
            handleHistorySearchOutside
          );
        };

      },
      []
    );


    async function chooseHistoryStock(
      symbol
    ) {

      const normalized =
        normalizeStockSymbol(
          symbol
        );


      if (
        !normalized
      ) {

        return;
      }


      setHistorySymbol(
        normalized
      );

      setHistoryDateRange(
        "ALL"
      );

      setHistorySearch(
        ""
      );

      setHistorySearchOpen(
        false
      );

      setHistoryTrackingMessage(
        "Tracking enabled"
      );


      try {

        await fetch(
          `${API_URL}/prediction-track/${encodeURIComponent(
            normalized
          )}`,
          {
            method:
              "POST",
          }
        );

      } catch {

        setHistoryTrackingMessage(
          "History loaded; auto-tracking will retry when backend is reachable."
        );
      }


      setTimeout(
        () => {

          historyDetailRef.current
            ?.scrollIntoView({
              behavior:
                "smooth",
              block:
                "start",
            });

        },
        120
      );
    }


    function submitHistorySearch() {

      const clean =
        historySearch
          .trim()
          .toLowerCase();


      if (
        !clean
      ) {

        setHistorySearchOpen(
          true
        );

        return;
      }


      const exact =
        allStocks.find(
          (
            item
          ) =>
            item.short
              .toLowerCase() ===
              clean
            ||
            item.symbol
              .toLowerCase() ===
              clean
            ||
            item.symbol
              .replace(
                ".NS",
                ""
              )
              .toLowerCase() ===
              clean
            ||
            item.name
              .toLowerCase() ===
              clean
        );


      const partial =
        exact ||
        historySearchMatches[
          0
        ];


      chooseHistoryStock(
        partial?.symbol ||
        clean
      );
    }


    async function refreshSelectedHistory() {

      try {

        setPredictionValidationLoading(
          true
        );

        setPredictionValidationError(
          ""
        );


        const response =
          await fetch(
            `${API_URL}/prediction-history-refresh/${encodeURIComponent(
              historySymbol
            )}`,
            {
              method:
                "POST",
            }
          );


        const data =
          await response
            .json()
            .catch(
              () =>
                null
            );


        if (
          !response.ok
        ) {

          throw new Error(
            data?.detail ||
            "Unable to refresh prediction history."
          );
        }


        setPredictionValidation(
          data
        );

        setHistoryTrackingMessage(
          "History checked and auto-tracking is enabled."
        );


      } catch (
        error
      ) {

        setPredictionValidationError(
          error.message ||
          "Unable to refresh prediction history."
        );


      } finally {

        setPredictionValidationLoading(
          false
        );
      }
    }


    useEffect(
      () => {

        setHistoryOverviewPage(
          1
        );

      },
      [
        historyOverviewSearch,
        historyOverviewStatus,
      ]
    );


    useEffect(
      () => {

        fetchFullNseCaptureStatus();


        const timer =
          setInterval(
            () => {

              fetchFullNseCaptureStatus();

            },
            10000
          );


        return () => {

          clearInterval(
            timer
          );
        };

      },
      []
    );


    useEffect(
      () => {

        const timer =
          setTimeout(
            () => {

              fetchHistoryOverview(
                historyOverviewPage,
                historyOverviewSearch,
                historyOverviewStatus
              );

            },
            250
          );


        return () => {

          clearTimeout(
            timer
          );
        };

      },
      [
        historyOverviewPage,
        historyOverviewSearch,
        historyOverviewStatus,
        predictionValidation
          ?.metrics
          ?.total_predictions,
      ]
    );


    const indicators =
      stock?.indicators ||
      {};


    const isGeneralizedUnseenPrediction =
      prediction?.coverage_mode ===
        "GENERALIZED_UNSEEN_STOCK" ||
      prediction?.symbol_validation_status ===
        "UNSEEN_EXPERIMENTAL";


    const forecastEngineLabel =
      isGeneralizedUnseenPrediction
        ? "Universal Next-Day Forecast"
        : "X2 Next-Day Forecast";


    const forecastPointLabel =
      isGeneralizedUnseenPrediction
        ? "Universal Expected Close"
        : "X2 Expected Close";


    const x2Point =
      prediction?.experimental_x2_point ||
      null;


    const x2Range =
      prediction?.expected_range ||
      null;


    const x2History =
      prediction?.historical_error_profile ||
      {};


    const displayedPredictionPrice =
      Number(
        x2Point?.price ??
        prediction?.predicted_price
      );


    const predictedMove =
      Number(
        x2Point?.move_percent ??
        prediction?.predicted_return_percent ??
        0
      );


    const predictedMoveRupees =
      Number(
        x2Point?.move_rupees ??
        (
          displayedPredictionPrice -
          Number(
            prediction?.current_close ||
            0
          )
        )
      );


    const x2RangeLower =
      Number(
        x2Range?.lower
      );


    const x2RangeUpper =
      Number(
        x2Range?.upper
      );


    const x2HoldoutCoverage =
      Number(
        x2Range?.holdout_observed_coverage_percent
      );


    const x2WalkCoverage =
      Number(
        x2Range?.walk_forward_observed_coverage_percent
      );


    const predictionSignal =
      prediction?.trend_signal ||
      (
        predictedMove >= 0.25
          ? "BULLISH"
          : predictedMove <= -0.25
          ? "BEARISH"
          : "NEUTRAL"
      );


    const isPositivePrediction =
      predictedMove >=
      0;


    const recentChart =
      Array.isArray(
        stock?.chart
      )
        ? stock.chart.slice(
            -22
          )
        : [];


    const recentPrices =
      recentChart
        .map(
          (
            point
          ) =>
            Number(
              point?.price
            )
        )
        .filter(
          (
            value
          ) =>
            Number.isFinite(
              value
            )
        );


    const recentLow =
      recentPrices.length
        ? Math.min(
            ...recentPrices
          )
        : null;


    const recentHigh =
      recentPrices.length
        ? Math.max(
            ...recentPrices
          )
        : null;


    const predictionChartData =
      recentChart.map(
        (
          point,
          index
        ) => ({
          ...point,
          historicalPrice:
            Number(
              point?.price
            ),
          predictedPrice:
            index ===
            recentChart.length -
              1
              ? Number(
                  point?.price
                )
              : null,
        })
      );


    if (
      prediction &&
      Number.isFinite(
        displayedPredictionPrice
      )
    ) {

      predictionChartData.push({
        time:
          "Next",
        historicalPrice:
          null,
        predictedPrice:
          displayedPredictionPrice,
      });
    }


    const rsi =
      Number(
        indicators?.rsi14
      );


    const macd =
      Number(
        indicators?.macd
      );


    const macdSignal =
      Number(
        indicators?.macd_signal
      );


    const sma20 =
      Number(
        indicators?.sma20
      );


    const ema20 =
      Number(
        indicators?.ema20
      );


    const volatility =
      Number(
        indicators?.volatility20
      );


    const volatilityLabel =
      !Number.isFinite(
        volatility
      )
        ? "--"
        : volatility >=
          2
        ? "High"
        : volatility >=
          1
        ? "Medium"
        : "Low";


    const rsiLabel =
      !Number.isFinite(
        rsi
      )
        ? "--"
        : rsi >=
          70
        ? "Overbought"
        : rsi <=
          30
        ? "Oversold"
        : rsi >=
          50
        ? "Positive"
        : "Neutral";


    const macdLabel =
      !Number.isFinite(
        macd
      )
        ? "--"
        : macd >=
          0
        ? "Bullish"
        : "Bearish";


    const priceVsSma =
      Number.isFinite(
        sma20
      ) &&
      Number.isFinite(
        Number(
          stock?.price
        )
      )
        ? Number(
            stock.price
          ) >=
          sma20
          ? "Above SMA"
          : "Below SMA"
        : "--";


    const priceVsEma =
      Number.isFinite(
        ema20
      ) &&
      Number.isFinite(
        Number(
          stock?.price
        )
      )
        ? Number(
            stock.price
          ) >=
          ema20
          ? "Above EMA"
          : "Below EMA"
        : "--";


    const relativeProbabilities =
      relativePrediction?.probabilities ||
      {};


    const rawUnderperform =
      Number(
        relativeProbabilities?.underperform ||
        0
      );


    const rawNeutral =
      Number(
        relativeProbabilities?.neutral ||
        0
      );


    const rawOutperform =
      Number(
        relativeProbabilities?.outperform ||
        0
      );


    const topRawScore =
      Number(
        relativePrediction?.top_probability ||
        0
      );


    const currentPrice =
      Number(
        stock?.price
      );


    const dayChange =
      Number(
        stock?.change_percent
      );


    return (
      <div>

        {/* =================================================
            STOCK HEADER
        ================================================= */}

        <div className="flex flex-col justify-between gap-4 border-b border-white/5 pb-4 xl:flex-row xl:items-center">

          <div className="flex min-w-0 items-center gap-4">

            <button
              onClick={() =>
                setActivePage(
                  "Dashboard"
                )
              }
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-white/5 bg-[#0f141d] text-gray-500 transition hover:text-white"
              title="Back to dashboard"
            >
              ←
            </button>


            <div className="min-w-0">

              <div className="flex flex-wrap items-center gap-3">

                <div>

                  <h1 className="truncate text-xl font-bold text-white">
                    {selectedStockInfo.short}
                  </h1>

                  <p className="mt-0.5 truncate text-[10px] uppercase tracking-wide text-gray-600">
                    {selectedStockInfo.name} · NSE
                  </p>

                </div>


                <div className="hidden h-9 w-px bg-white/5 sm:block" />


                <div>

                  <p className="text-xl font-bold text-white">
                    {formatPrice(
                      currentPrice
                    )}
                  </p>

                  <p
                    className={`mt-0.5 text-xs font-semibold ${
                      dayChange >= 0
                        ? "text-green-400"
                        : "text-red-400"
                    }`}
                  >
                    {formatPercent(
                      dayChange
                    )}
                    {" "}
                    <span className="font-normal text-gray-600">
                      Today
                    </span>
                  </p>

                </div>

              </div>

            </div>

          </div>


          <div className="flex flex-wrap items-center gap-2">

            <button
              onClick={() =>
                toggleWatchlist(
                  selectedSymbol
                )
              }
              className={`flex items-center gap-2 rounded-xl border px-4 py-2.5 text-xs font-medium transition ${
                isWatchlisted(
                  selectedSymbol
                )
                  ? "border-yellow-500/20 bg-yellow-500/10 text-yellow-400"
                  : "border-blue-500/30 bg-blue-500/5 text-blue-400 hover:bg-blue-500/10"
              }`}
            >

              <Star
                size={15}
                fill={
                  isWatchlisted(
                    selectedSymbol
                  )
                    ? "currentColor"
                    : "none"
                }
              />

              {isWatchlisted(
                selectedSymbol
              )
                ? "In Watchlist"
                : "Add to Watchlist"}

            </button>


            <button
              onClick={() => {

                fetchPrediction(
                  selectedSymbol
                );

                fetchRelativePrediction(
                  selectedSymbol
                );

              }}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-500 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-violet-500/15 transition hover:brightness-110"
            >

              <Activity
                size={14}
              />

              Refresh AI

            </button>

          </div>

        </div>


        {/* =================================================
            TOP PREDICTION SECTION
        ================================================= */}

        <div className="mt-5 grid gap-4 xl:grid-cols-[1.08fr_0.92fr]">

          {/* AI PRICE PREDICTION */}

          <div className="rounded-2xl border border-[#1b2738] bg-[#0f141d] p-5">

            <div className="flex items-center gap-3">

              <div className="rounded-xl bg-violet-500/10 p-2.5">

                <Bot
                  size={19}
                  className="text-violet-400"
                />

              </div>


              <div>

                <h2 className="text-sm font-semibold text-white">
                  {forecastEngineLabel}
                </h2>

                <p className="mt-1 text-[10px] text-gray-600">
                  {isGeneralizedUnseenPrediction
                    ? "Generalized experimental NSE forecast · stock-agnostic features"
                    : "Probabilistic next-day forecast · uncertainty-aware"}
                </p>

              </div>

            </div>


            {predictionLoading ? (

              <div className="mt-5">

                <LoadingBox
                  text={
                    isGeneralizedUnseenPrediction
                      ? "Running universal forecast..."
                      : "Running X2 forecast..."
                  }
                />

              </div>

            ) : predictionError ? (

              <div className="mt-5 rounded-xl border border-red-500/20 bg-red-500/10 p-4">

                <p className="text-xs font-medium text-red-400">
                  Prediction unavailable
                </p>

                <p className="mt-2 text-[10px] leading-5 text-red-300/70">
                  {predictionError}
                </p>

              </div>

            ) : prediction ? (

              <>

                <div className="mt-5 grid gap-3 md:grid-cols-2">

                  <div className="rounded-xl border border-violet-500/10 bg-[linear-gradient(145deg,rgba(124,58,237,0.08),rgba(11,16,24,0.65))] p-4">

                    <div className="flex items-center justify-between gap-3">

                      <p className="text-[10px] text-gray-500">
                        {forecastPointLabel}
                      </p>

                      <span className="rounded-full border border-yellow-500/15 bg-yellow-500/10 px-2 py-1 text-[8px] font-semibold text-yellow-400">
                        {isGeneralizedUnseenPrediction
                          ? "UNSEEN STOCK · EXPERIMENTAL"
                          : "EXPERIMENTAL POINT"}
                      </span>

                    </div>

                    <p
                      className={`mt-2 text-3xl font-bold ${
                        isPositivePrediction
                          ? "text-green-400"
                          : "text-red-400"
                      }`}
                    >
                      {Number.isFinite(
                        displayedPredictionPrice
                      )
                        ? formatPrice(
                            displayedPredictionPrice
                          )
                        : "--"}
                    </p>

                    <p
                      className={`mt-2 text-xs font-semibold ${
                        isPositivePrediction
                          ? "text-green-400"
                          : "text-red-400"
                      }`}
                    >
                      {Number.isFinite(
                        predictedMoveRupees
                      )
                        ? `${predictedMoveRupees >= 0 ? "+" : "-"}${formatPrice(
                            Math.abs(
                              predictedMoveRupees
                            )
                          )}`
                        : "--"}
                      {" "}
                      ({formatPercent(
                        predictedMove,
                        4
                      )})
                    </p>

                    <p className="mt-3 text-[9px] leading-4 text-gray-600">
                      {isGeneralizedUnseenPrediction
                        ? "This stock was not part of the original universal training universe. The forecast uses stock-agnostic technical and market features, so treat it as experimental generalization."
                        : "The exact X2 point is shown for transparency. The backend keeps the production central estimate baseline-safe until the point model beats its baseline."}
                    </p>

                  </div>


                  <div className="rounded-xl border border-blue-500/15 bg-[#0b1018] p-4">

                    <div className="flex items-center justify-between gap-3">

                      <p className="text-[10px] text-gray-500">
                        80% Expected Range
                      </p>

                      <span className="rounded-full border border-green-500/15 bg-green-500/10 px-2 py-1 text-[8px] font-semibold text-green-400">
                        {isGeneralizedUnseenPrediction
                          ? "GLOBAL VALIDATION BAND"
                          : "EMPIRICALLY HEALTHY"}
                      </span>

                    </div>

                    <p className="mt-3 text-xl font-bold text-white">
                      {Number.isFinite(
                        x2RangeLower
                      ) && Number.isFinite(
                        x2RangeUpper
                      )
                        ? `${formatPrice(
                            x2RangeLower
                          )} – ${formatPrice(
                            x2RangeUpper
                          )}`
                        : "--"}
                    </p>

                    <div className="mt-4 grid grid-cols-2 gap-2">

                      <div className="rounded-lg bg-white/[0.03] p-2.5">
                        <p className="text-[8px] text-gray-600">
                          Holdout Coverage
                        </p>
                        <p className="mt-1 text-xs font-bold text-green-400">
                          {Number.isFinite(
                            x2HoldoutCoverage
                          )
                            ? `${x2HoldoutCoverage.toFixed(
                                2
                              )}%`
                            : "--"}
                        </p>
                      </div>

                      <div className="rounded-lg bg-white/[0.03] p-2.5">
                        <p className="text-[8px] text-gray-600">
                          Walk-Forward
                        </p>
                        <p className="mt-1 text-xs font-bold text-green-400">
                          {Number.isFinite(
                            x2WalkCoverage
                          )
                            ? `${x2WalkCoverage.toFixed(
                                2
                              )}%`
                            : "--"}
                        </p>
                      </div>

                    </div>

                    <div className="mt-3 flex items-center justify-between gap-3 border-t border-white/5 pt-3">
                      <span className="text-[9px] text-gray-600">
                        {isGeneralizedUnseenPrediction
                          ? "Universal signal"
                          : "X2 signal"}
                      </span>
                      <TrendBadge signal={predictionSignal} />
                    </div>

                  </div>

                </div>


                <div className="mt-3 grid gap-2 sm:grid-cols-4">

                  {[
                    {
                      label: "Median Error",
                      value:
                        x2History?.median_error_rupees !== null &&
                        x2History?.median_error_rupees !== undefined
                          ? formatPrice(
                              x2History.median_error_rupees
                            )
                          : "--",
                    },
                    {
                      label: "Within ₹10",
                      value:
                        x2History?.within_10_percent !== null &&
                        x2History?.within_10_percent !== undefined
                          ? `${Number(
                              x2History.within_10_percent
                            ).toFixed(
                              2
                            )}%`
                          : "--",
                    },
                    {
                      label: "Within ₹20",
                      value:
                        x2History?.within_20_percent !== null &&
                        x2History?.within_20_percent !== undefined
                          ? `${Number(
                              x2History.within_20_percent
                            ).toFixed(
                              2
                            )}%`
                          : "--",
                    },
                    {
                      label: "Over ₹30",
                      value:
                        x2History?.over_30_percent !== null &&
                        x2History?.over_30_percent !== undefined
                          ? `${Number(
                              x2History.over_30_percent
                            ).toFixed(
                              2
                            )}%`
                          : "--",
                    },
                  ].map(
                    (
                      item
                    ) => (
                      <div
                        key={item.label}
                        className="rounded-lg border border-white/5 bg-[#0b1018] px-3 py-2.5"
                      >
                        <p className="text-[8px] text-gray-600">
                          {item.label}
                        </p>
                        <p className="mt-1 text-xs font-semibold text-gray-200">
                          {item.value}
                        </p>
                      </div>
                    )
                  )}

                </div>


                <div
                className={
                  historyChartExpanded
                    ? "fixed inset-0 z-[120] flex h-screen w-screen flex-col overflow-hidden bg-[#070b11] p-5 md:p-7"
                    : "mt-4 rounded-xl border border-white/5 bg-[#0b1018] p-4"
                }
              >

                  <div className="flex flex-wrap items-center justify-between gap-3">

                    <h3 className="text-xs font-semibold text-white">
                      Historical Price + Next-Day Forecast
                    </h3>


                    <div className="flex items-center gap-4 text-[9px] text-gray-500">

                      <span className="flex items-center gap-1.5">

                        <span className="h-0.5 w-5 bg-blue-400" />

                        Historical

                      </span>


                      <span className="flex items-center gap-1.5">

                        <span className="h-0.5 w-5 border-t border-dashed border-green-400" />

                        AI Forecast

                      </span>

                    </div>

                  </div>


                  <div className="mt-3 h-[245px]">

                    <ResponsiveContainer
                      width="100%"
                      height="100%"
                    >

                      <ComposedChart
                        data={
                          predictionChartData
                        }
                        margin={{
                          top: 10,
                          right: 8,
                          bottom: 0,
                          left: 0,
                        }}
                      >

                        <CartesianGrid
                          stroke="#1d2531"
                          strokeDasharray="3 3"
                          vertical={false}
                        />


                        <XAxis
                          dataKey="time"
                          axisLine={false}
                          tickLine={false}
                          minTickGap={35}
                          tick={{
                            fill:
                              "#64748b",
                            fontSize:
                              9,
                          }}
                        />


                        <YAxis
                          orientation="right"
                          axisLine={false}
                          tickLine={false}
                          width={46}
                          domain={[
                            "auto",
                            "auto",
                          ]}
                          tick={{
                            fill:
                              "#64748b",
                            fontSize:
                              9,
                          }}
                          tickFormatter={
                            (
                              value
                            ) =>
                              Number(
                                value
                              ).toFixed(
                                0
                              )
                          }
                        />


                        <Tooltip
                          content={
                            <MarketChartTooltip />
                          }
                        />


                        <Line
                          type="monotone"
                          dataKey="historicalPrice"
                          stroke="#3b82f6"
                          strokeWidth={2}
                          dot={false}
                          connectNulls={false}
                          isAnimationActive={
                            false
                          }
                        />


                        <Line
                          type="monotone"
                          dataKey="predictedPrice"
                          stroke="#4ade80"
                          strokeWidth={2}
                          strokeDasharray="6 5"
                          dot={{
                            r:
                              3,
                            fill:
                              "#4ade80",
                          }}
                          connectNulls
                          isAnimationActive={
                            false
                          }
                        />

                      </ComposedChart>

                    </ResponsiveContainer>

                  </div>


                  <p className="mt-2 text-[9px] leading-4 text-gray-600">
                    {isGeneralizedUnseenPrediction
                      ? "Blue = recent market history. Green dashed point = generalized universal experimental estimate; use the expected range for uncertainty context."
                      : "Blue = recent market history. Green dashed point = experimental X2 central estimate; use the 80% range above for uncertainty context."}
                  </p>

                </div>

              </>

            ) : (

              <div className="mt-5 rounded-xl bg-white/[0.03] p-5 text-xs text-gray-500">
                No AI prediction available.
              </div>

            )}

          </div>


          {/* AI PREDICTION SUMMARY */}

          <div className="rounded-2xl border border-[#1b2738] bg-[#0f141d] p-5">

            <div className="flex items-center gap-3">

              <div className="rounded-xl bg-blue-500/10 p-2.5">

                <Gauge
                  size={19}
                  className="text-blue-400"
                />

              </div>


              <div>

                <h2 className="text-sm font-semibold text-white">
                  AI Prediction Summary
                </h2>

                <p className="mt-1 text-[10px] text-gray-600">
                  {isGeneralizedUnseenPrediction
                  ? "Universal price estimate, live market context and V9."
                  : "X2 price range, experimental point estimate, live market context and V9."}
                </p>

              </div>

            </div>


            <div className="mt-5 overflow-hidden rounded-xl border border-white/5">

              {[
                {
                  label:
                    "Trend Direction",
                  sub:
                    "X2 next-day signal",
                  value:
                    predictionSignal,
                  tone:
                    predictionSignal ===
                    "BULLISH"
                      ? "text-green-400"
                      : predictionSignal ===
                        "BEARISH"
                      ? "text-red-400"
                      : "text-yellow-400",
                  icon:
                    predictionSignal ===
                    "BEARISH"
                      ? TrendingDown
                      : TrendingUp,
                },
                {
                  label:
                    "Latest Daily Close",
                  sub:
                    "Model reference price",
                  value:
                    prediction
                      ? formatPrice(
                          prediction.current_close
                        )
                      : "--",
                  tone:
                    "text-white",
                  icon:
                    Activity,
                },
                {
                  label:
                    "X2 Expected Close",
                  sub:
                    "Experimental central estimate",
                  value:
                    Number.isFinite(
                      displayedPredictionPrice
                    )
                      ? formatPrice(
                          displayedPredictionPrice
                        )
                      : "--",
                  tone:
                    isPositivePrediction
                      ? "text-green-400"
                      : "text-red-400",
                  icon:
                    Bot,
                },
                {
                  label:
                    "80% Expected Range",
                  sub:
                    "Empirically calibrated X2 interval",
                  value:
                    Number.isFinite(
                      x2RangeLower
                    ) && Number.isFinite(
                      x2RangeUpper
                    )
                      ? `${formatPrice(
                          x2RangeLower
                        )} – ${formatPrice(
                          x2RangeUpper
                        )}`
                      : "--",
                  tone:
                    "text-blue-300",
                  icon:
                    Gauge,
                },
                {
                  label:
                    "Recent Price Range",
                  sub:
                    "Low – high from visible recent history",
                  value:
                    recentLow !==
                      null &&
                    recentHigh !==
                      null
                      ? `${formatPrice(
                          recentLow
                        )} – ${formatPrice(
                          recentHigh
                        )}`
                      : "--",
                  tone:
                    "text-gray-200",
                  icon:
                    BarChart3,
                },
                {
                  label:
                    "20-Day Volatility",
                  sub:
                    Number.isFinite(
                      volatility
                    )
                      ? `${volatility.toFixed(
                          2
                        )}% daily std. dev.`
                      : "Indicator unavailable",
                  value:
                    volatilityLabel,
                  tone:
                    volatilityLabel ===
                    "High"
                      ? "text-red-400"
                      : volatilityLabel ===
                        "Medium"
                      ? "text-yellow-400"
                      : volatilityLabel ===
                        "Low"
                      ? "text-green-400"
                      : "text-gray-500",
                  icon:
                    Gauge,
                },
                {
                  label:
                    "V9 Relative Outlook",
                  sub:
                    "5-day performance vs NIFTY 50",
                  value:
                    relativePrediction?.signal ||
                    "--",
                  tone:
                    String(
                      relativePrediction?.signal ||
                      ""
                    ).toUpperCase() ===
                    "OUTPERFORM"
                      ? "text-green-400"
                      : String(
                          relativePrediction?.signal ||
                          ""
                        ).toUpperCase() ===
                        "UNDERPERFORM"
                      ? "text-red-400"
                      : "text-yellow-400",
                  icon:
                    Star,
                },
              ].map(
                (
                  item
                ) => {

                  const SummaryIcon =
                    item.icon;


                  return (
                    <div
                      key={
                        item.label
                      }
                      className="flex items-center justify-between gap-4 border-b border-white/5 bg-[#0b1018] px-4 py-4 last:border-0"
                    >

                      <div className="flex min-w-0 items-center gap-3">

                        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white/[0.035]">

                          <SummaryIcon
                            size={14}
                            className="text-gray-500"
                          />

                        </div>


                        <div className="min-w-0">

                          <p className="text-[11px] font-medium text-gray-300">
                            {item.label}
                          </p>

                          <p className="mt-0.5 truncate text-[9px] text-gray-600">
                            {item.sub}
                          </p>

                        </div>

                      </div>


                      <p className={`shrink-0 text-xs font-semibold ${item.tone}`}>
                        {item.value}
                      </p>

                    </div>
                  );
                }
              )}

            </div>


            <div className="mt-4 rounded-xl border border-blue-500/10 bg-blue-500/[0.04] p-4">

              <p className="text-[10px] leading-5 text-gray-500">
                X2 provides an experimental central estimate plus an empirical 80% price range. V9 answers a separate 5-day relative-strength question. Neither is a guaranteed trading signal.
              </p>

            </div>

          </div>

        </div>


        {/* =================================================
            LOWER INFORMATION CARDS
        ================================================= */}

        <div className="mt-4 grid gap-4 xl:grid-cols-3">

          {/* TECHNICAL INDICATORS */}

          <div className="rounded-2xl border border-[#1b2738] bg-[#0f141d] p-5">

            <div className="flex items-center gap-3">

              <BarChart3
                size={18}
                className="text-gray-400"
              />

              <h2 className="text-sm font-semibold text-white">
                Technical Indicators
              </h2>

            </div>


            <div className="mt-4">

              {[
                {
                  label:
                    "RSI (14)",
                  value:
                    Number.isFinite(
                      rsi
                    )
                      ? rsi.toFixed(
                          2
                        )
                      : "--",
                  status:
                    rsiLabel,
                  tone:
                    rsi >=
                    70
                      ? "text-red-400"
                      : rsi <=
                        30
                      ? "text-blue-400"
                      : rsi >=
                        50
                      ? "text-green-400"
                      : "text-yellow-400",
                },
                {
                  label:
                    "MACD",
                  value:
                    Number.isFinite(
                      macd
                    )
                      ? macd.toFixed(
                          4
                        )
                      : "--",
                  status:
                    macdLabel,
                  tone:
                    macd >=
                    0
                      ? "text-green-400"
                      : "text-red-400",
                },
                {
                  label:
                    "MACD Signal",
                  value:
                    Number.isFinite(
                      macdSignal
                    )
                      ? macdSignal.toFixed(
                          4
                        )
                      : "--",
                  status:
                    "Signal line",
                  tone:
                    "text-gray-300",
                },
                {
                  label:
                    "SMA (20)",
                  value:
                    Number.isFinite(
                      sma20
                    )
                      ? formatPrice(
                          sma20
                        )
                      : "--",
                  status:
                    priceVsSma,
                  tone:
                    priceVsSma ===
                    "Above SMA"
                      ? "text-green-400"
                      : "text-red-400",
                },
                {
                  label:
                    "EMA (20)",
                  value:
                    Number.isFinite(
                      ema20
                    )
                      ? formatPrice(
                          ema20
                        )
                      : "--",
                  status:
                    priceVsEma,
                  tone:
                    priceVsEma ===
                    "Above EMA"
                      ? "text-green-400"
                      : "text-red-400",
                },
              ].map(
                (
                  item
                ) => (

                  <div
                    key={
                      item.label
                    }
                    className="flex items-center justify-between gap-3 border-b border-white/5 py-3 last:border-0"
                  >

                    <span className="text-[10px] text-gray-500">
                      {item.label}
                    </span>


                    <div className="text-right">

                      <p className="text-[11px] font-medium text-gray-300">
                        {item.value}
                      </p>

                      <p className={`mt-0.5 text-[9px] font-medium ${item.tone}`}>
                        {item.status}
                      </p>

                    </div>

                  </div>

                )
              )}

            </div>

          </div>


          {/* V9 RAW SCORES */}

          <div className="rounded-2xl border border-[#1b2738] bg-[#0f141d] p-5">

            <div className="flex items-center gap-3">

              <Gauge
                size={18}
                className="text-violet-400"
              />

              <div>

                <h2 className="text-sm font-semibold text-white">
                  V9 Relative Strength Scores
                </h2>

                <p className="mt-1 text-[9px] text-gray-600">
                  Raw classifier outputs · not calibrated confidence
                </p>

              </div>

            </div>


            {relativeLoading ? (

              <div className="mt-5">

                <LoadingBox
                  text="Running V9 analysis..."
                />

              </div>

            ) : relativeError ? (

              <div className="mt-5 rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-[10px] text-red-400">
                {relativeError}
              </div>

            ) : (

              <>

                <div className="mt-5 flex items-center justify-center">

                  <div className="relative flex h-32 w-32 items-center justify-center rounded-full bg-[conic-gradient(#22c55e_0_var(--outperform),#eab308_var(--outperform)_var(--neutral),#ef4444_var(--neutral)_100%)]"
                    style={{
                      "--outperform":
                        `${Math.max(
                          0,
                          Math.min(
                            100,
                            rawOutperform *
                              100
                          )
                        )}%`,
                      "--neutral":
                        `${Math.max(
                          0,
                          Math.min(
                            100,
                            (
                              rawOutperform +
                              rawNeutral
                            ) *
                              100
                          )
                        )}%`,
                    }}
                  >

                    <div className="flex h-[94px] w-[94px] flex-col items-center justify-center rounded-full bg-[#0f141d]">

                      <p className="text-xl font-bold text-white">
                        {formatProbabilityScore(
                          topRawScore
                        )}
                      </p>

                      <p className="mt-1 text-[8px] text-gray-600">
                        Top raw score
                      </p>

                    </div>

                  </div>

                </div>


                <div className="mt-5 space-y-3">

                  {[
                    {
                      label:
                        "Outperform",
                      value:
                        rawOutperform,
                      color:
                        "bg-green-400",
                      text:
                        "text-green-400",
                    },
                    {
                      label:
                        "Neutral",
                      value:
                        rawNeutral,
                      color:
                        "bg-yellow-400",
                      text:
                        "text-yellow-400",
                    },
                    {
                      label:
                        "Underperform",
                      value:
                        rawUnderperform,
                      color:
                        "bg-red-400",
                      text:
                        "text-red-400",
                    },
                  ].map(
                    (
                      item
                    ) => (

                      <div
                        key={
                          item.label
                        }
                      >

                        <div className="mb-1 flex items-center justify-between">

                          <span className="text-[10px] text-gray-500">
                            {item.label}
                          </span>

                          <span className={`text-[10px] font-semibold ${item.text}`}>
                            {formatProbabilityScore(
                              item.value
                            )}
                          </span>

                        </div>


                        <div className="h-1.5 overflow-hidden rounded-full bg-white/5">

                          <div
                            className={`h-full rounded-full ${item.color}`}
                            style={{
                              width:
                                `${Math.max(
                                  0,
                                  Math.min(
                                    100,
                                    item.value *
                                      100
                                  )
                                )}%`,
                            }}
                          />

                        </div>

                      </div>

                    )
                  )}

                </div>

              </>

            )}

          </div>


          {/* MODEL INFORMATION */}

          <div className="rounded-2xl border border-[#1b2738] bg-[#0f141d] p-5">

            <div className="flex items-center gap-3">

              <Bot
                size={18}
                className="text-blue-400"
              />

              <h2 className="text-sm font-semibold text-white">
                AI Model Information
              </h2>

            </div>


            <div className="mt-4">

              {[
                {
                  label:
                    "Price Forecast Model",
                  value:
                    prediction?.model ||
                    "BiLSTM",
                },
                {
                  label:
                    "Forecast Horizon",
                  value:
                    "Next trading day",
                },
                {
                  label:
                    "Input Window",
                  value:
                    "60 trading days",
                },
                {
                  label:
                    "Engineered Inputs",
                  value:
                    "11 features",
                },
                {
                  label:
                    "Relative Model",
                  value:
                    "V9 · 5D vs NIFTY 50",
                },
                {
                  label:
                    "V9 Features",
                  value:
                    "29 features",
                },
              ].map(
                (
                  item
                ) => (

                  <div
                    key={
                      item.label
                    }
                    className="flex items-center justify-between gap-4 border-b border-white/5 py-3 last:border-0"
                  >

                    <span className="text-[10px] text-gray-500">
                      {item.label}
                    </span>

                    <span className="text-right text-[10px] font-medium text-gray-300">
                      {item.value}
                    </span>

                  </div>

                )
              )}

            </div>


            <div className="mt-4 rounded-xl border border-green-500/10 bg-green-500/[0.04] p-3">

              <div className="flex items-center gap-2">

                <CheckCircle2
                  size={14}
                  className="text-green-400"
                />

                <p className="text-[10px] font-medium text-green-400">
                  V9 walk-forward evaluation available
                </p>

              </div>


              <p className="mt-2 text-[9px] leading-4 text-gray-600">
                V9 evaluation is shown separately because training accuracy is not used as final performance evidence.
              </p>

            </div>

          </div>

        </div>


        {/* =================================================
            V9 EVALUATION STRIP
        ================================================= */}

        {relativePrediction?.walk_forward_evaluation && (

          <div className="mt-4 rounded-2xl border border-[#1b2738] bg-[#0f141d] p-5">

            <div className="flex flex-col justify-between gap-3 md:flex-row md:items-center">

              <div>

                <h2 className="text-sm font-semibold text-white">
                  V9 Walk-Forward Evaluation
                </h2>

                <p className="mt-1 text-[10px] text-gray-600">
                  Historical future-only evaluation · separate from the live raw score above.
                </p>

              </div>


              <RelativeStrengthBadge
                signal={
                  relativePrediction.signal
                }
              />

            </div>


            <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">

              <StatCard
                title="Accuracy"
                value={
                  relativePrediction
                    .walk_forward_evaluation
                    .accuracy_percent !==
                  null
                    ? `${Number(
                        relativePrediction
                          .walk_forward_evaluation
                          .accuracy_percent
                      ).toFixed(
                        2
                      )}%`
                    : "--"
                }
                subtitle="3-class target"
              />


              <StatCard
                title="Macro F1"
                value={
                  relativePrediction
                    .walk_forward_evaluation
                    .macro_f1_percent !==
                  null
                    ? `${Number(
                        relativePrediction
                          .walk_forward_evaluation
                          .macro_f1_percent
                      ).toFixed(
                        2
                      )}%`
                    : "--"
                }
                subtitle="All classes"
              />


              <StatCard
                title="Balanced Accuracy"
                value={
                  relativePrediction
                    .walk_forward_evaluation
                    .balanced_accuracy_percent !==
                  null
                    ? `${Number(
                        relativePrediction
                          .walk_forward_evaluation
                          .balanced_accuracy_percent
                      ).toFixed(
                        2
                      )}%`
                    : "--"
                }
                subtitle="Walk-forward"
              />


              <StatCard
                title="vs Majority"
                value={
                  relativePrediction
                    .walk_forward_evaluation
                    .improvement_vs_majority_pp !==
                  null
                    ? `+${Number(
                        relativePrediction
                          .walk_forward_evaluation
                          .improvement_vs_majority_pp
                      ).toFixed(
                        2
                      )} pp`
                    : "--"
                }
                subtitle="Baseline"
              />


              <StatCard
                title="vs Momentum"
                value={
                  relativePrediction
                    .walk_forward_evaluation
                    .improvement_vs_momentum_pp !==
                  null
                    ? `+${Number(
                        relativePrediction
                          .walk_forward_evaluation
                          .improvement_vs_momentum_pp
                      ).toFixed(
                        2
                      )} pp`
                    : "--"
                }
                subtitle="Baseline"
              />

            </div>

          </div>

        )}


        {/* =================================================
            LIVE PREDICTION VALIDATION
        ================================================= */}

        <div className="mt-4 grid min-w-0 gap-4 xl:grid-cols-[1.55fr_0.95fr]">

          {/* LEFT: HISTORY TABLE */}

          <div className="min-w-0 max-w-full overflow-hidden rounded-2xl border border-violet-500/20 bg-[#0f141d] p-4 shadow-[0_0_0_1px_rgba(124,58,237,0.05)]">

            <div className="border-b border-white/5 pb-4">

              <div className="flex min-w-0 flex-col justify-between gap-3 xl:flex-row xl:items-start">

                <div className="min-w-0">

                  <h2 className="text-sm font-semibold text-white">
                    All-Time Prediction History
                  </h2>

                  <p className="mt-1 text-[10px] text-gray-600">
                    Saved forecast vs actual close for every captured trading date of{" "}
                    <span className="font-semibold text-violet-300">
                      {historyStockInfo.short}
                    </span>.
                  </p>

                </div>


                <div className="flex shrink-0 flex-wrap items-center gap-2">

                  <span className="rounded-full border border-green-500/15 bg-green-500/[0.06] px-2.5 py-1 text-[9px] font-medium text-green-400">
                    ● Auto Daily Tracking
                  </span>


                  <button
                    onClick={
                      refreshSelectedHistory
                    }
                    disabled={
                      predictionValidationLoading
                    }
                    className="w-fit rounded-lg border border-white/10 bg-[#0b1018] px-3 py-2 text-[10px] font-medium text-gray-300 transition hover:border-violet-500/30 hover:text-violet-400 disabled:cursor-wait disabled:opacity-60"
                  >
                    {predictionValidationLoading
                      ? "Checking..."
                      : "Refresh History"}
                  </button>

                </div>

              </div>


              <div
                ref={
                  historySearchRef
                }
                className="relative mt-4"
              >

                <div className="flex w-full items-center rounded-xl border border-white/10 bg-[#0b1018] transition focus-within:border-violet-500/40">

                  <Search
                    size={16}
                    className="ml-3 shrink-0 text-violet-400"
                  />


                  <input
                    value={
                      historySearch
                    }
                    onFocus={() =>
                      setHistorySearchOpen(
                        true
                      )
                    }
                    onChange={
                      (
                        event
                      ) => {

                        setHistorySearch(
                          event.target.value
                        );

                        setHistorySearchOpen(
                          true
                        );
                      }
                    }
                    onKeyDown={
                      (
                        event
                      ) => {

                        if (
                          event.key ===
                          "Enter"
                        ) {

                          event.preventDefault();

                          submitHistorySearch();
                        }


                        if (
                          event.key ===
                          "Escape"
                        ) {

                          setHistorySearchOpen(
                            false
                          );
                        }
                      }
                    }
                    placeholder="Search prediction history: RELIANCE, JIOFIN, TCS, INFY..."
                    className="min-w-0 flex-1 bg-transparent px-3 py-3 text-xs text-white outline-none placeholder:text-gray-700"
                  />


                  <button
                    onClick={
                      submitHistorySearch
                    }
                    className="mr-1 rounded-lg bg-violet-500 px-3 py-2 text-[10px] font-semibold text-white transition hover:bg-violet-400"
                  >
                    Search
                  </button>

                </div>


                {historySearchOpen && (

                  <div className="absolute left-0 right-0 top-[calc(100%+6px)] z-50 max-h-72 overflow-y-auto rounded-xl border border-white/10 bg-[#0b1018] p-1.5 shadow-2xl shadow-black/50">

                    {historySearchMatches.length >
                    0 ? (

                      historySearchMatches.map(
                        (
                          item
                        ) => (

                          <button
                            key={
                              item.symbol
                            }
                            onClick={() =>
                              chooseHistoryStock(
                                item.symbol
                              )
                            }
                            className="flex w-full items-center justify-between gap-3 rounded-lg px-3 py-2.5 text-left transition hover:bg-white/[0.04]"
                          >

                            <div className="min-w-0">

                              <p className="truncate text-[11px] font-semibold text-gray-200">
                                {item.short}
                              </p>

                              <p className="mt-0.5 truncate text-[9px] text-gray-600">
                                {item.name}
                              </p>

                            </div>


                            <span className="shrink-0 text-[9px] text-gray-700">
                              {item.symbol}
                            </span>

                          </button>

                        )
                      )

                    ) : (

                      <button
                        onClick={
                          submitHistorySearch
                        }
                        className="w-full rounded-lg px-3 py-3 text-left text-[10px] text-gray-500 transition hover:bg-white/[0.04]"
                      >
                        Track custom NSE symbol:{" "}
                        <span className="font-semibold text-violet-300">
                          {historySearch
                            .trim()
                            .toUpperCase()}
                        </span>
                      </button>

                    )}

                  </div>

                )}

              </div>


              <div className="mt-3 flex flex-wrap items-center gap-2 text-[9px]">

                <span className="rounded-full border border-white/5 bg-white/[0.02] px-2.5 py-1 text-gray-500">
                  Viewing:{" "}
                  <strong className="font-semibold text-white">
                    {historyStockInfo.short}
                  </strong>
                </span>


                {predictionValidation?.tracking?.tracked && (

                  <span className="rounded-full border border-blue-500/10 bg-blue-500/[0.04] px-2.5 py-1 text-blue-400">
                    ✓ Saved for future daily history
                  </span>

                )}


                {historyTrackingMessage && (

                  <span className="text-gray-600">
                    {historyTrackingMessage}
                  </span>

                )}

              </div>

            </div>


            <div className="mt-4 rounded-2xl border border-violet-500/15 bg-violet-500/[0.035] p-4">

              <div className="flex flex-col justify-between gap-4 xl:flex-row xl:items-center">

                <div className="min-w-0">

                  <p className="text-[9px] font-semibold uppercase tracking-[0.14em] text-violet-400">
                    Full NSE Daily Capture
                  </p>

                  <h3 className="mt-1 text-sm font-semibold text-white">
                    Build prediction history for the entire NSE universe
                  </h3>

                  <p className="mt-1 max-w-3xl text-[9px] leading-4 text-gray-600">
                    StockVision now attempts every listed NSE symbol after the completed session. It runs in a rate-limited background batch so you do not need to search stocks manually.
                  </p>

                </div>


                <button
                  onClick={
                    startFullNseCapture
                  }
                  disabled={
                    fullNseCaptureLoading ||
                    fullNseCapture?.running
                  }
                  className="w-fit rounded-lg border border-violet-500/25 bg-violet-500/10 px-4 py-2.5 text-[10px] font-semibold text-violet-300 transition hover:bg-violet-500/20 disabled:cursor-wait disabled:opacity-50"
                >
                  {fullNseCapture?.running
                    ? "Full NSE Capture Running"
                    : fullNseCaptureLoading
                    ? "Starting..."
                    : "Start / Resume Full NSE Capture"}
                </button>

              </div>


              <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-8">

                {[
                  {
                    label:
                      "Session",
                    value:
                      fullNseCapture?.session_date ||
                      "--",
                  },
                  {
                    label:
                      "Total",
                    value:
                      fullNseCapture?.total ??
                      "--",
                  },
                  {
                    label:
                      "Processed",
                    value:
                      fullNseCapture?.processed ??
                      "--",
                  },
                  {
                    label:
                      "Captured",
                    value:
                      fullNseCapture?.captured ??
                      "--",
                  },
                  {
                    label:
                      "Already Saved",
                    value:
                      fullNseCapture?.already_saved ??
                      "--",
                  },
                  {
                    label:
                      "No Data",
                    value:
                      fullNseCapture?.data_unavailable ??
                      "--",
                  },
                  {
                    label:
                      "Model Stale",
                    value:
                      fullNseCapture?.model_data_stale ??
                      "--",
                  },
                  {
                    label:
                      "Failed",
                    value:
                      fullNseCapture?.failed ??
                      "--",
                  },
                ].map(
                  (
                    item
                  ) => (

                    <div
                      key={
                        item.label
                      }
                      className="rounded-lg border border-white/5 bg-[#080d14] px-3 py-2.5"
                    >

                      <p className="text-[8px] uppercase tracking-wide text-gray-700">
                        {item.label}
                      </p>

                      <p className="mt-1 text-xs font-bold text-white">
                        {item.value}
                      </p>

                    </div>

                  )
                )}

              </div>


              <div className="mt-3">

                <div className="mb-1.5 flex items-center justify-between text-[8px] text-gray-600">

                  <span>
                    {fullNseCapture?.last_symbol
                      ? `Last: ${fullNseCapture.last_symbol} · ${fullNseCapture.last_status || ""}`
                      : "Waiting to begin"}
                  </span>

                  <span>
                    {Number(
                      fullNseCapture?.progress_percent ||
                      0
                    ).toFixed(
                      2
                    )}%
                  </span>

                </div>


                <div className="h-2 overflow-hidden rounded-full bg-white/[0.04]">

                  <div
                    className="h-full rounded-full bg-gradient-to-r from-violet-500 via-blue-500 to-green-400 transition-all duration-500"
                    style={{
                      width:
                        `${Math.min(
                          100,
                          Math.max(
                            0,
                            Number(
                              fullNseCapture?.progress_percent ||
                              0
                            )
                          )
                        )}%`,
                    }}
                  />

                </div>

              </div>


              <p className="mt-3 text-[8px] leading-4 text-gray-700">
                This tries the full NSE list automatically. A few securities may still remain unavailable if Yahoo has no usable history, the listing is too new, or the model cannot form enough features. Those are shown honestly instead of receiving fabricated predictions.
              </p>

            </div>


            <div className="mt-4 rounded-2xl border border-blue-500/10 bg-[#0d131c] p-4">

              <div className="flex flex-col justify-between gap-3 lg:flex-row lg:items-center">

                <div>

                  <h3 className="text-xs font-semibold text-white">
                    All NSE Stock Prediction History
                  </h3>

                  <p className="mt-1 text-[9px] text-gray-600">
                    Every NSE stock is visible here, even if you have never searched it before.
                  </p>

                </div>


                <div className="flex flex-wrap items-center gap-2">

                  <div className="flex min-w-[230px] items-center rounded-lg border border-white/10 bg-[#080d14] px-3">

                    <Search
                      size={13}
                      className="shrink-0 text-blue-400"
                    />

                    <input
                      value={
                        historyOverviewSearch
                      }
                      onChange={
                        (
                          event
                        ) =>
                          setHistoryOverviewSearch(
                            event.target.value
                          )
                      }
                      placeholder="Search all stocks..."
                      className="min-w-0 flex-1 bg-transparent px-2 py-2.5 text-[10px] text-white outline-none placeholder:text-gray-700"
                    />

                  </div>


                  <select
                    value={
                      historyOverviewStatus
                    }
                    onChange={
                      (
                        event
                      ) =>
                        setHistoryOverviewStatus(
                          event.target.value
                        )
                    }
                    className="rounded-lg border border-white/10 bg-[#080d14] px-3 py-2.5 text-[10px] text-gray-300 outline-none"
                  >
                    <option value="ALL">
                      All Stocks
                    </option>

                    <option value="WITH_HISTORY">
                      With History
                    </option>

                    <option value="RESOLVED">
                      Latest Resolved
                    </option>

                    <option value="PENDING">
                      Latest Pending
                    </option>

                    <option value="NO_HISTORY">
                      No History Yet
                    </option>
                  </select>

                </div>

              </div>


              <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-6">

                {[
                  {
                    label:
                      "NSE Stocks",
                    value:
                      historyOverview
                        ?.summary
                        ?.total_nse_stocks ??
                      "--",
                  },
                  {
                    label:
                      "With History",
                    value:
                      historyOverview
                        ?.summary
                        ?.stocks_with_saved_history ??
                      "--",
                  },
                  {
                    label:
                      "No History",
                    value:
                      historyOverview
                        ?.summary
                        ?.stocks_without_saved_history ??
                      "--",
                  },
                  {
                    label:
                      "Saved Predictions",
                    value:
                      historyOverview
                        ?.summary
                        ?.total_saved_predictions ??
                      "--",
                  },
                  {
                    label:
                      "Resolved",
                    value:
                      historyOverview
                        ?.summary
                        ?.resolved_predictions ??
                      "--",
                  },
                  {
                    label:
                      "Pending",
                    value:
                      historyOverview
                        ?.summary
                        ?.pending_predictions ??
                      "--",
                  },
                ].map(
                  (
                    item
                  ) => (

                    <div
                      key={
                        item.label
                      }
                      className="rounded-lg border border-white/5 bg-[#080d14] px-3 py-2.5"
                    >

                      <p className="text-[8px] uppercase tracking-wide text-gray-700">
                        {item.label}
                      </p>

                      <p className="mt-1 text-sm font-bold text-white">
                        {item.value}
                      </p>

                    </div>

                  )
                )}

              </div>


              {historyOverviewError ? (

                <div className="mt-4 rounded-lg border border-red-500/15 bg-red-500/[0.04] px-3 py-2 text-[10px] text-red-400">
                  {historyOverviewError}
                </div>

              ) : (

                <div className="mt-4 overflow-x-auto">

                  <table className="w-full min-w-[1320px]">

                    <thead>

                      <tr className="border-b border-white/5 bg-[#080d14] text-left text-[8px] uppercase tracking-wide text-gray-600">

                        <th className="px-3 py-2.5">
                          Stock
                        </th>

                        <th className="px-3 py-2.5">
                          Records
                        </th>

                        <th className="px-3 py-2.5">
                          Prediction Date
                        </th>

                        <th className="px-3 py-2.5">
                          Predicted Value
                        </th>

                        <th className="px-3 py-2.5">
                          Actual Value
                        </th>

                        <th className="px-3 py-2.5">
                          Difference
                        </th>

                        <th className="px-3 py-2.5">
                          Final Value
                        </th>

                        <th className="px-3 py-2.5">
                          Abs Error
                        </th>

                        <th className="px-3 py-2.5">
                          Status
                        </th>

                        <th className="sticky right-0 z-10 border-l border-white/5 bg-[#080d14] px-3 py-2.5 text-right">
                          History
                        </th>

                      </tr>

                    </thead>


                    <tbody>

                      {historyOverviewLoading &&
                      !historyOverview ? (

                        <tr>

                          <td
                            colSpan={10}
                            className="px-3 py-8 text-center text-[10px] text-gray-600"
                          >
                            Loading all NSE stocks...
                          </td>

                        </tr>

                      ) : (
                        historyOverview
                          ?.rows ||
                        []
                      ).length >
                      0 ? (

                        (
                          historyOverview
                            ?.rows ||
                          []
                        ).map(
                          (
                            row
                          ) => {

                            const difference =
                              row.latest_difference;


                            return (
                              <tr
                                key={
                                  row.symbol
                                }
                                className="border-b border-white/[0.035] text-[9px] transition hover:bg-white/[0.02]"
                              >

                                <td className="px-3 py-2.5">

                                  <p className="font-semibold text-gray-200">
                                    {row.short}
                                  </p>

                                  <p className="mt-0.5 max-w-[180px] truncate text-[8px] text-gray-700">
                                    {row.name}
                                  </p>

                                </td>


                                <td className="px-3 py-2.5 text-gray-400">
                                  {row.saved_records}
                                </td>


                                <td className="px-3 py-2.5 text-gray-400">
                                  {row.latest_prediction_date ||
                                    "--"}
                                </td>


                                <td className="px-3 py-2.5 font-medium text-blue-300">
                                  {row.latest_forecast !==
                                  null &&
                                  row.latest_forecast !==
                                  undefined
                                    ? formatPrice(
                                        row.latest_forecast
                                      )
                                    : "--"}
                                </td>


                                <td className="px-3 py-2.5 font-medium text-green-300">
                                  {row.latest_actual_close !==
                                  null &&
                                  row.latest_actual_close !==
                                  undefined
                                    ? formatPrice(
                                        row.latest_actual_close
                                      )
                                    : "--"}
                                </td>


                                <td
                                  className={`px-3 py-2.5 font-semibold ${
                                    difference ===
                                      null ||
                                    difference ===
                                      undefined
                                      ? "text-gray-700"
                                      : Math.abs(
                                          Number(
                                            difference
                                          )
                                        ) <=
                                        20
                                      ? "text-green-400"
                                      : Math.abs(
                                          Number(
                                            difference
                                          )
                                        ) <=
                                        30
                                      ? "text-amber-400"
                                      : "text-red-400"
                                  }`}
                                >
                                  {difference !==
                                  null &&
                                  difference !==
                                  undefined
                                    ? `${Number(
                                        difference
                                      ) >=
                                      0
                                        ? "+"
                                        : "-"}${formatPrice(
                                        Math.abs(
                                          Number(
                                            difference
                                          )
                                        )
                                      )}`
                                    : "--"}
                                </td>


                                <td className="px-3 py-2.5 font-medium text-violet-300">
                                  {row.latest_final_value !==
                                  null &&
                                  row.latest_final_value !==
                                  undefined
                                    ? formatPrice(
                                        row.latest_final_value
                                      )
                                    : "--"}
                                </td>


                                <td className="px-3 py-2.5 text-gray-400">
                                  {row.latest_absolute_error !==
                                  null &&
                                  row.latest_absolute_error !==
                                  undefined
                                    ? formatPrice(
                                        row.latest_absolute_error
                                      )
                                    : "--"}
                                </td>


                                <td className="px-3 py-2.5">

                                  <span
                                    className={`inline-flex rounded-full px-2 py-1 text-[8px] font-semibold ${
                                      row.latest_status ===
                                      "RESOLVED"
                                        ? "bg-green-500/10 text-green-400"
                                        : row.latest_status ===
                                          "PENDING"
                                        ? "bg-amber-500/10 text-amber-400"
                                        : "bg-white/[0.04] text-gray-600"
                                    }`}
                                  >
                                    {row.latest_status ===
                                    "NO_HISTORY"
                                      ? fullNseCapture?.running
                                        ? "QUEUED / NO HISTORY"
                                        : "NO HISTORY"
                                      : row.latest_status}
                                  </span>

                                </td>


                                <td className="sticky right-0 border-l border-white/5 bg-[#0d131c] px-3 py-2.5 text-right">

                                  <button
                                    onClick={() =>
                                      chooseHistoryStock(
                                        row.symbol
                                      )
                                    }
                                    className="rounded-lg border border-violet-500/20 bg-violet-500/[0.07] px-2.5 py-1.5 text-[8px] font-semibold text-violet-300 transition hover:bg-violet-500/15"
                                  >
                                    View Full History
                                  </button>

                                </td>

                              </tr>
                            );
                          }
                        )

                      ) : (

                        <tr>

                          <td
                            colSpan={10}
                            className="px-3 py-8 text-center text-[10px] text-gray-600"
                          >
                            No stocks match this filter.
                          </td>

                        </tr>

                      )}

                    </tbody>

                  </table>

                </div>

              )}


              <div className="mt-3 flex flex-col justify-between gap-2 sm:flex-row sm:items-center">

                <p className="text-[8px] leading-4 text-gray-700">
                  All NSE stocks are visible. “No History” means no real live forecast has been captured yet; StockVision does not invent older predictions.
                </p>


                <div className="flex items-center gap-2">

                  <button
                    disabled={
                      (
                        historyOverview
                          ?.pagination
                          ?.page ||
                        1
                      ) <=
                      1
                    }
                    onClick={() =>
                      setHistoryOverviewPage(
                        (
                          value
                        ) =>
                          Math.max(
                            1,
                            value -
                              1
                          )
                      )
                    }
                    className="rounded-lg border border-white/5 bg-[#080d14] px-2.5 py-1.5 text-[9px] text-gray-400 disabled:opacity-30"
                  >
                    Previous
                  </button>

                  <span className="text-[9px] text-gray-600">
                    Page{" "}
                    {historyOverview
                      ?.pagination
                      ?.page ||
                      1}
                    {" "}of{" "}
                    {historyOverview
                      ?.pagination
                      ?.total_pages ||
                      1}
                  </span>

                  <button
                    disabled={
                      (
                        historyOverview
                          ?.pagination
                          ?.page ||
                        1
                      ) >=
                      (
                        historyOverview
                          ?.pagination
                          ?.total_pages ||
                        1
                      )
                    }
                    onClick={() =>
                      setHistoryOverviewPage(
                        (
                          value
                        ) =>
                          value +
                          1
                      )
                    }
                    className="rounded-lg border border-white/5 bg-[#080d14] px-2.5 py-1.5 text-[9px] text-gray-400 disabled:opacity-30"
                  >
                    Next
                  </button>

                </div>

              </div>

            </div>


            <div
              ref={
                historyDetailRef
              }
              className="scroll-mt-24"
            />


            <div className="mt-4 rounded-xl border border-violet-500/15 bg-violet-500/[0.035] px-4 py-3">

              <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">

                <div>

                  <p className="text-[9px] font-semibold uppercase tracking-[0.14em] text-violet-400">
                    Full Date History
                  </p>

                  <h3 className="mt-1 text-sm font-semibold text-white">
                    {historyStockInfo.short} — Complete Prediction History
                  </h3>

                  <p className="mt-1 text-[9px] leading-4 text-gray-600">
                    Every saved prediction date for this stock is shown below. Resolved rows contain the actual next-trading-day close and the prediction difference.
                  </p>

                </div>


                <div className="flex flex-wrap items-center gap-2">

                  <span className="rounded-full border border-white/5 bg-[#0b1018] px-3 py-1.5 text-[9px] text-gray-400">
                    {predictionValidation?.history?.length ||
                      0} total saved dates
                  </span>

                  <span className="rounded-full border border-green-500/10 bg-green-500/[0.05] px-3 py-1.5 text-[9px] text-green-400">
                    {predictionValidation?.metrics
                      ?.resolved_predictions ||
                      0} resolved
                  </span>

                  <span className="rounded-full border border-amber-500/10 bg-amber-500/[0.05] px-3 py-1.5 text-[9px] text-amber-400">
                    {predictionValidation?.metrics
                      ?.pending_predictions ||
                      0} pending
                  </span>

                </div>

              </div>

            </div>


            {Array.isArray(
              predictionValidation?.history
            ) &&
            predictionValidation.history.length >
              0 && (

              <div className="mt-4 rounded-2xl border border-emerald-500/10 bg-[#0b1118] p-4">

                <div className="flex flex-col justify-between gap-3 lg:flex-row lg:items-center">

                  <div>

                    <p className="text-[9px] font-semibold uppercase tracking-[0.14em] text-emerald-400">
                      Prediction Performance
                    </p>

                    <h3 className="mt-1 text-sm font-semibold text-white">
                      Proper Model Evaluation — {historyStockInfo.short}
                    </h3>

                    <p className="mt-1 max-w-3xl text-[9px] leading-4 text-gray-600">
                      Calculated only from RESOLVED predictions in the selected {historyDateRange === "ALL" ? "All Dates" : historyDateRange} range. Pending predictions are excluded. LIVE and historical REPLAY performance are kept separate.
                    </p>

                  </div>


                  <div className="rounded-lg border border-white/5 bg-[#080d14] px-3 py-2 text-[8px] leading-4 text-gray-500">
                    Lower MAE / Error % is better · Higher Direction / Range % is better
                  </div>

                </div>


                {[
                  {
                    key:
                      "live",
                    title:
                      "LIVE Performance",
                    subtitle:
                      "Real forecasts actually captured by StockVision",
                    data:
                      historyPerformance.live,
                    accent:
                      "text-green-400",
                    border:
                      "border-green-500/10",
                    background:
                      "bg-green-500/[0.025]",
                  },
                  {
                    key:
                      "replay",
                    title:
                      "REPLAY / Backtest Performance",
                    subtitle:
                      "Historical model replay — useful for backtesting, not live proof",
                    data:
                      historyPerformance.replay,
                    accent:
                      "text-blue-400",
                    border:
                      "border-blue-500/10",
                    background:
                      "bg-blue-500/[0.025]",
                  },
                ].map(
                  (
                    section
                  ) => {

                    const metrics =
                      section.data;


                    const improvement =
                      metrics
                        ?.baselineImprovement;


                    const hasSamples =
                      (
                        metrics
                          ?.priceSamples ||
                        0
                      ) >
                      0;


                    return (

                      <div
                        key={
                          section.key
                        }
                        className={`mt-4 rounded-xl border ${section.border} ${section.background} p-4`}
                      >

                        <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-center">

                          <div>

                            <h4 className={`text-xs font-semibold ${section.accent}`}>
                              {section.title}
                            </h4>

                            <p className="mt-1 text-[8px] text-gray-600">
                              {section.subtitle}
                            </p>

                          </div>


                          <span className="w-fit rounded-full border border-white/5 bg-[#080d14] px-2.5 py-1 text-[8px] text-gray-500">
                            {metrics?.resolvedRows ||
                              0} resolved samples
                          </span>

                        </div>


                        {hasSamples ? (

                          <>

                            <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7">

                              {[
                                {
                                  label:
                                    "Direction Accuracy",
                                  value:
                                    metrics.directionAccuracy !==
                                    null
                                      ? `${metrics.directionAccuracy.toFixed(
                                          2
                                        )}%`
                                      : "--",
                                  note:
                                    `${metrics.directionSamples || 0} directional samples`,
                                },
                                {
                                  label:
                                    "MAE",
                                  value:
                                    metrics.mae !==
                                    null
                                      ? formatPrice(
                                          metrics.mae
                                        )
                                      : "--",
                                  note:
                                    "Avg absolute price error",
                                },
                                {
                                  label:
                                    "Avg Error",
                                  value:
                                    metrics.mape !==
                                    null
                                      ? `${metrics.mape.toFixed(
                                          2
                                        )}%`
                                      : "--",
                                  note:
                                    "Mean percentage error",
                                },
                                {
                                  label:
                                    "Range Coverage",
                                  value:
                                    metrics.rangeCoverage !==
                                    null
                                      ? `${metrics.rangeCoverage.toFixed(
                                          2
                                        )}%`
                                      : "--",
                                  note:
                                    `${metrics.rangeSamples || 0} range samples`,
                                },
                                {
                                  label:
                                    "Baseline MAE",
                                  value:
                                    metrics.baselineMae !==
                                    null
                                      ? formatPrice(
                                          metrics.baselineMae
                                        )
                                      : "--",
                                  note:
                                    "Previous-close baseline",
                                },
                                {
                                  label:
                                    "Vs Baseline",
                                  value:
                                    improvement !==
                                    null
                                      ? `${improvement >=
                                        0
                                          ? "+"
                                          : ""}${improvement.toFixed(
                                          2
                                        )}%`
                                      : "--",
                                  note:
                                    improvement ===
                                    null
                                      ? "Not enough data"
                                      : improvement >
                                        0
                                      ? "Model MAE is better"
                                      : improvement <
                                        0
                                      ? "Baseline MAE is better"
                                      : "Same MAE",
                                  valueClass:
                                    improvement ===
                                    null
                                      ? "text-gray-400"
                                      : improvement >
                                        0
                                      ? "text-green-400"
                                      : improvement <
                                        0
                                      ? "text-red-400"
                                      : "text-gray-300",
                                },
                                {
                                  label:
                                    "Resolved",
                                  value:
                                    metrics.resolvedRows,
                                  note:
                                    "Pending excluded",
                                },
                              ].map(
                                (
                                  card
                                ) => (

                                  <div
                                    key={
                                      card.label
                                    }
                                    className="rounded-lg border border-white/5 bg-[#080d14] px-3 py-3"
                                  >

                                    <p className="text-[8px] uppercase tracking-wide text-gray-700">
                                      {card.label}
                                    </p>

                                    <p className={`mt-1.5 text-base font-bold ${
                                      card.valueClass ||
                                      "text-white"
                                    }`}>
                                      {card.value}
                                    </p>

                                    <p className="mt-1 text-[8px] leading-3 text-gray-700">
                                      {card.note}
                                    </p>

                                  </div>

                                )
                              )}

                            </div>


                            <div className="mt-3 rounded-lg border border-white/5 bg-[#080d14] px-3 py-2.5 text-[9px] leading-4">

                              {section.key ===
                              "live" &&
                              metrics.resolvedRows <
                                20 ? (

                                <span className="text-amber-300">
                                  Live sample is still small ({metrics.resolvedRows}). Treat these numbers as early evidence; they become more meaningful as automatic daily history grows.
                                </span>

                              ) : improvement !==
                                null &&
                                improvement >
                                  0 ? (

                                <span className="text-green-300">
                                  On these resolved samples, the model's MAE is {Math.abs(
                                    improvement
                                  ).toFixed(
                                    2
                                  )}% lower than the previous-close baseline.
                                </span>

                              ) : improvement !==
                                null &&
                                improvement <
                                  0 ? (

                                <span className="text-red-300">
                                  On these resolved samples, the previous-close baseline is currently {Math.abs(
                                    improvement
                                  ).toFixed(
                                    2
                                  )}% better in MAE. This should be reported honestly rather than calling the model more accurate.
                                </span>

                              ) : (

                                <span className="text-gray-500">
                                  More resolved samples are required for a meaningful baseline comparison.
                                </span>

                              )}

                            </div>

                          </>

                        ) : (

                          <div className="mt-3 rounded-lg border border-white/5 bg-[#080d14] px-4 py-5 text-center">

                            <p className="text-[10px] font-medium text-gray-400">
                              No resolved {section.key === "live" ? "LIVE" : "REPLAY"} samples in this date range yet.
                            </p>

                            <p className="mt-1 text-[8px] text-gray-700">
                              Pending rows are never included in performance metrics.
                            </p>

                          </div>

                        )}

                      </div>

                    );
                  }
                )}


                <div className="mt-3 rounded-lg border border-violet-500/10 bg-violet-500/[0.03] px-3 py-2.5 text-[8px] leading-4 text-gray-600">
                  <span className="font-semibold text-violet-300">
                    Baseline:
                  </span>{" "}
                  predicts the next close as the current/base close. A positive “Vs Baseline” means StockVision has lower MAE on that sample; a negative value means the simple baseline performed better.
                </div>

              </div>

            )}


            {Array.isArray(
              predictionValidation?.history
            ) &&
            predictionValidation.history.length >
              0 && (

              <div className="mt-4 rounded-xl border border-white/5 bg-[#0b1018] p-4">

                <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">

                  <div>

                    <div className="flex items-center gap-2">

                      <LineChartIcon
                        size={15}
                        className="text-violet-400"
                      />

                      <h3 className="text-xs font-semibold text-white">
                        {historyDateRange ===
                        "ALL"
                          ? "All-Date Prediction History Chart"
                          : `${historyDateRange} Prediction History Chart`}
                      </h3>

                    </div>

                    <div className="mt-1 flex flex-wrap items-center gap-2">

                      <p className="text-[9px] text-gray-600">
                      {historyDateRange ===
                      "ALL"
                        ? historyChartExpanded
                          ? "Full-screen view: all saved dates with predicted value, actual value, difference and final value. Press Esc to exit."
                          : "All saved dates are plotted: predicted value, actual value, difference and final value. Use Full Screen for a wide detailed view."
                        : `Showing ${historyDateRange} only. Click All Dates below to show the complete saved history.`}
                      </p>

                      {predictionValidationLoading && (

                        <span className="inline-flex items-center gap-1 rounded-full border border-white/5 bg-white/[0.025] px-2 py-1 text-[8px] text-gray-600">
                          <LoaderCircle
                            size={10}
                            className="animate-spin"
                          />
                          Updating data
                        </span>

                      )}

                    </div>

                  </div>


                  <div className="flex flex-wrap items-center gap-3 text-[9px]">

                    <span className="flex items-center gap-1.5 text-blue-300">
                      <span className="h-2 w-2 rounded-full bg-blue-400" />
                      Predicted Value
                    </span>

                    <span className="flex items-center gap-1.5 text-green-300">
                      <span className="h-2 w-2 rounded-full bg-green-400" />
                      Actual Value
                    </span>

                    <span className="flex items-center gap-1.5 text-amber-300">
                      <span className="h-2 w-2 rounded-full bg-amber-400" />
                      Difference
                    </span>

                    <span className="flex items-center gap-1.5 text-violet-300">
                      <span className="h-2 w-2 rounded-full bg-violet-400" />
                      Final Value
                    </span>

                    <span className="flex items-center gap-1.5 text-sky-300">
                      <span className="h-[2px] w-4 bg-sky-400" />
                      80% Range
                    </span>


                    <button
                      type="button"
                      onClick={() =>
                        setHistoryChartExpanded(
                          (
                            value
                          ) =>
                            !value
                        )
                      }
                      className="ml-1 inline-flex items-center gap-1.5 rounded-lg border border-violet-500/20 bg-violet-500/[0.06] px-3 py-2 text-[9px] font-semibold text-violet-300 transition hover:bg-violet-500/10"
                      title={
                        historyChartExpanded
                          ? "Exit full screen"
                          : "Open full screen chart"
                      }
                    >

                      {historyChartExpanded ? (
                        <Minimize2
                          size={13}
                        />
                      ) : (
                        <Maximize2
                          size={13}
                        />
                      )}

                      {historyChartExpanded
                        ? "Exit Full Screen"
                        : "Full Screen"}

                    </button>

                  </div>

                </div>


                <div
                  className={
                    historyChartExpanded
                      ? "mt-5 min-h-0 flex-1 w-full"
                      : "mt-4 h-[440px] w-full xl:h-[500px]"
                  }
                >

                  <ResponsiveContainer
                    width="100%"
                    height="100%"
                    debounce={120}
                  >

                    <ComposedChart
                      data={
                        historyChartData
                      }
                      margin={{
                        top:
                          8,
                        right:
                          historyChartExpanded
                            ? 36
                            : 16,
                        left:
                          historyChartExpanded
                            ? 8
                            : -12,
                        bottom:
                          0,
                      }}
                    >

                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#1f2937"
                        vertical={false}
                      />


                      <XAxis
                        dataKey="date"
                        axisLine={false}
                        tickLine={false}
                        minTickGap={24}
                        tick={{
                          fill:
                            "#64748b",
                          fontSize:
                            historyChartExpanded
                              ? 12
                              : 9,
                        }}
                      />


                      <YAxis
                        yAxisId="price"
                        orientation="right"
                        axisLine={false}
                        tickLine={false}
                        width={52}
                        domain={[
                          "auto",
                          "auto",
                        ]}
                        tick={{
                          fill:
                            "#64748b",
                          fontSize:
                            historyChartExpanded
                              ? 12
                              : 9,
                        }}
                        tickFormatter={
                          (
                            value
                          ) =>
                            `₹${Number(
                              value
                            ).toFixed(
                              0
                            )}`
                        }
                      />


                      <YAxis
                        yAxisId="difference"
                        orientation="left"
                        axisLine={false}
                        tickLine={false}
                        width={48}
                        domain={[
                          "auto",
                          "auto",
                        ]}
                        tick={{
                          fill:
                            "#64748b",
                          fontSize:
                            historyChartExpanded
                              ? 12
                              : 9,
                        }}
                        tickFormatter={
                          (
                            value
                          ) =>
                            `${Number(
                              value
                            ).toFixed(
                              0
                            )}`
                        }
                      />


                      <Tooltip
                        contentStyle={{
                          background:
                            "#0f141d",
                          border:
                            "1px solid #273244",
                          borderRadius:
                            "10px",
                          color:
                            "#e5e7eb",
                          fontSize:
                            "10px",
                        }}
                        formatter={
                          (
                            value,
                            name
                          ) => {

                            if (
                              value ===
                              null ||
                              value ===
                              undefined
                            ) {

                              return [
                                "--",
                                name,
                              ];
                            }


                            return [
                              formatPrice(
                                Number(
                                  value
                                )
                              ),
                              name,
                            ];
                          }
                        }
                      />


                      <Line
                        type="monotone"
                        yAxisId="price"
                        dataKey="rangeLow"
                        name="80% Range Low"
                        stroke="#60a5fa"
                        strokeWidth={1.5}
                        strokeDasharray="4 4"
                        dot={{
                          r:
                            2.5,
                          fill:
                            "#60a5fa",
                        }}
                        connectNulls
                        isAnimationActive={
                          false
                        }
                      />


                      <Line
                        type="monotone"
                        yAxisId="price"
                        dataKey="rangeHigh"
                        name="80% Range High"
                        stroke="#60a5fa"
                        strokeWidth={1.5}
                        strokeDasharray="4 4"
                        dot={{
                          r:
                            2.5,
                          fill:
                            "#60a5fa",
                        }}
                        connectNulls
                        isAnimationActive={
                          false
                        }
                      />


                      <Line
                        yAxisId="price"
                        type="monotone"
                        dataKey="predictedValue"
                        name="Predicted Value"
                        stroke="#3b82f6"
                        strokeWidth={2.2}
                        dot={{
                          r:
                            3,
                          fill:
                            "#3b82f6",
                        }}
                        activeDot={{
                          r:
                            5,
                        }}
                        connectNulls
                        isAnimationActive={
                          false
                        }
                      />


                      <Line
                        yAxisId="price"
                        type="monotone"
                        dataKey="actualValue"
                        name="Actual Value"
                        stroke="#22c55e"
                        strokeWidth={2.2}
                        dot={{
                          r:
                            3,
                          fill:
                            "#22c55e",
                        }}
                        activeDot={{
                          r:
                            5,
                        }}
                        connectNulls
                        isAnimationActive={
                          false
                        }
                      />


                      <Line
                        yAxisId="difference"
                        type="monotone"
                        dataKey="difference"
                        name="Difference"
                        stroke="#f59e0b"
                        strokeWidth={1.8}
                        strokeDasharray="5 4"
                        dot={{
                          r:
                            2.5,
                          fill:
                            "#f59e0b",
                        }}
                        connectNulls
                        isAnimationActive={
                          false
                        }
                      />


                      <Line
                        yAxisId="price"
                        type="monotone"
                        dataKey="finalValue"
                        name="Final Value"
                        stroke="#a855f7"
                        strokeWidth={1.8}
                        dot={{
                          r:
                            2.5,
                          fill:
                            "#a855f7",
                        }}
                        connectNulls
                        isAnimationActive={
                          false
                        }
                      />

                    </ComposedChart>

                  </ResponsiveContainer>

                </div>


                {!historyChartExpanded &&
                filteredPredictionHistory.length < 2 && (

                  <div className="mt-3 rounded-lg border border-amber-500/10 bg-amber-500/[0.04] px-3 py-2 text-[9px] text-amber-300">
                    {historyDateRange ===
                    "ALL"
                      ? "Only 1 saved prediction exists for this stock right now. New trading-date forecasts will be added permanently to this stock's history."
                      : `Only ${filteredPredictionHistory.length} saved prediction record(s) exist in the selected ${historyDateRange} range.`}
                  </div>

                )}


                {!historyChartExpanded && (

                  <p className="mt-2 text-[9px] leading-4 text-gray-600">
                    Pending rows keep the original saved forecast. After the next completed NSE trading-day close is available, StockVision stores the actual close and forecast difference without rewriting the old prediction.
                  </p>

                )}

              </div>

            )}


            {predictionValidationError ? (

              <div className="mt-4 rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-xs text-red-400">
                {predictionValidationError}
              </div>

            ) : Array.isArray(
                predictionValidation?.history
              ) &&
              predictionValidation.history.length >
                0 ? (

              <>

              <div className="mt-5">

                <div className="mb-3 flex flex-col justify-between gap-2 sm:flex-row sm:items-center">

                  <div>

                    <h3 className="text-xs font-semibold text-white">
                      Prediction History — {historyStockInfo.short}
                    </h3>

                    <p className="mt-1 text-[9px] text-gray-600">
                      {historyDateRange ===
                      "ALL"
                        ? "Showing every available date: genuine LIVE captures plus clearly-labelled historical REPLAY rows, with predicted value, actual value and difference."
                        : `Showing ${historyDateRange} of saved prediction history. Click All Dates to restore the complete history.`}
                    </p>

                  </div>


                  <div className="flex flex-wrap items-center gap-1 rounded-lg border border-white/5 bg-[#0b1018] p-1">

                    {[
                      "7D",
                      "30D",
                      "3M",
                      "6M",
                      "1Y",
                      "ALL",
                    ].map(
                      (
                        range
                      ) => {

                        const active =
                          historyDateRange ===
                          range;


                        return (
                          <button
                            key={
                              range
                            }
                            onClick={() =>
                              setHistoryDateRange(
                                range
                              )
                            }
                            className={`rounded-md px-2.5 py-1.5 text-[9px] font-semibold transition ${
                              active
                                ? "bg-blue-500 text-white shadow-sm shadow-blue-500/20"
                                : "text-gray-500 hover:bg-white/[0.04] hover:text-gray-300"
                            }`}
                          >
                            {range ===
                            "ALL"
                              ? "All Dates"
                              : range}
                          </button>
                        );
                      }
                    )}

                  </div>

                </div>


              <div className="flex flex-wrap items-center justify-between gap-2">

                <div className="flex flex-wrap gap-2">

                  <span className="rounded-full border border-white/5 bg-[#0b1018] px-3 py-1 text-[9px] text-gray-500">
                    {historyDateRange ===
                    "ALL"
                      ? `${predictionValidation?.metrics?.total_predictions || predictionValidation.history.length} saved`
                      : `${filteredPredictionHistory.length} shown / ${predictionValidation.history.length} total`}
                  </span>

                  <span className="rounded-full border border-green-500/10 bg-green-500/[0.05] px-3 py-1 text-[9px] text-green-400">
                    {predictionValidation?.metrics
                      ?.resolved_predictions ||
                      0} resolved
                  </span>

                  <span className="rounded-full border border-amber-500/10 bg-amber-500/[0.05] px-3 py-1 text-[9px] text-amber-400">
                    {predictionValidation?.metrics
                      ?.pending_predictions ||
                      0} pending
                  </span>

                </div>


                <p className="text-[9px] text-gray-600">
                  {historyDateRange ===
                  "ALL"
                    ? (
                        predictionValidation?.history_start_date
                          ? `All saved dates from ${predictionValidation.history_start_date}`
                          : "History starts after the first saved live prediction"
                      )
                    : `${historyDateRange} history filter`}
                </p>

              </div>

              </div>


              <div className="mt-4 overflow-hidden rounded-xl border border-white/5 bg-[#0a0f17]">

                <div className="border-b border-white/5 bg-[#0b111a] px-4 py-3">

                  <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-center">

                    <div>

                      <p className="text-[9px] font-semibold uppercase tracking-[0.14em] text-violet-400">
                        Price Comparison
                      </p>

                      <h4 className="mt-1 text-xs font-semibold text-white">
                        Predicted Price vs Actual Price
                      </h4>

                      <p className="mt-1 text-[9px] text-gray-600">
                        Difference = Actual Price − Predicted Price
                      </p>

                    </div>


                    <div className="flex flex-wrap items-center gap-2 text-[8px]">

                      <span className="rounded-full border border-violet-500/15 bg-violet-500/[0.05] px-2.5 py-1 text-violet-300">
                        Predicted
                      </span>

                      <span className="rounded-full border border-green-500/15 bg-green-500/[0.05] px-2.5 py-1 text-green-300">
                        Actual
                      </span>

                      <span className="rounded-full border border-blue-500/15 bg-blue-500/[0.05] px-2.5 py-1 text-blue-300">
                        Difference
                      </span>

                    </div>

                  </div>

                </div>


                <div className="stockvision-validation-scroll w-full max-w-full overflow-x-auto overscroll-x-contain">

                  <table className="w-full min-w-[1240px]">

                    <thead>

                      <tr className="border-b border-white/5 bg-[#080d14] text-left text-[9px] uppercase tracking-wide text-gray-600">

                        <th className="sticky left-0 z-20 min-w-[120px] border-r border-white/5 bg-[#080d14] px-4 py-3 font-medium">
                          Prediction Date
                        </th>

                        <th className="min-w-[82px] px-3 py-3 font-medium">
                          Source
                        </th>

                        <th className="min-w-[130px] px-3 py-3 font-medium text-violet-400">
                          Predicted Price
                        </th>

                        <th className="min-w-[130px] px-3 py-3 font-medium text-green-400">
                          Actual Price
                        </th>

                        <th className="min-w-[130px] px-3 py-3 font-medium text-blue-400">
                          Difference
                        </th>

                        <th className="min-w-[90px] px-3 py-3 font-medium">
                          Error %
                        </th>

                        <th className="min-w-[100px] px-3 py-3 font-medium">
                          Status
                        </th>

                        <th className="min-w-[125px] px-3 py-3 font-medium">
                          For Date
                        </th>

                        <th className="min-w-[180px] px-3 py-3 font-medium">
                          Expected Range
                        </th>

                        <th className="min-w-[110px] px-3 py-3 font-medium">
                          Direction
                        </th>

                        <th className="min-w-[95px] px-3 py-3 font-medium">
                          Range
                        </th>

                      </tr>

                    </thead>


                    <tbody>

                      {filteredPredictionHistory.length ===
                      0 ? (

                        <tr>

                          <td
                            colSpan={11}
                            className="px-4 py-10 text-center"
                          >

                            <p className="text-xs font-medium text-gray-400">
                              No saved prediction dates in this range
                            </p>

                            <button
                              onClick={() =>
                                setHistoryDateRange(
                                  "ALL"
                                )
                              }
                              className="mt-2 rounded-lg border border-blue-500/20 bg-blue-500/[0.05] px-3 py-1.5 text-[9px] font-semibold text-blue-400 transition hover:bg-blue-500/10"
                            >
                              Show All Dates
                            </button>

                          </td>

                        </tr>

                      ) : filteredPredictionHistory.map(
                        (
                          row,
                          index
                        ) => {

                          const toFiniteOrNull =
                            (
                              value
                            ) => {

                              if (
                                value === null ||
                                value === undefined ||
                                value === ""
                              ) {

                                return null;
                              }


                              const number =
                                Number(
                                  value
                                );


                              return Number.isFinite(
                                number
                              )
                                ? number
                                : null;
                            };


                          const resolved =
                            row.status ===
                            "RESOLVED";


                          const savedForecast =
                            toFiniteOrNull(
                              row.forecast_point_price ??
                              row.experimental_x2_point_price ??
                              row.predicted_price
                            );


                          const actualClose =
                            toFiniteOrNull(
                              row.actual_close
                            );


                          const signedDifference =
                            resolved &&
                            savedForecast !==
                              null &&
                            actualClose !==
                              null
                              ? (
                                  row.forecast_difference_actual_minus_prediction ??
                                  (
                                    actualClose -
                                    savedForecast
                                  )
                                )
                              : null;


                          const absoluteError =
                            resolved &&
                            savedForecast !==
                              null &&
                            actualClose !==
                              null
                              ? (
                                  row.forecast_absolute_error ??
                                  Math.abs(
                                    actualClose -
                                    savedForecast
                                  )
                                )
                              : null;


                          const errorPercent =
                            resolved &&
                            actualClose !==
                              null &&
                            actualClose !==
                              0 &&
                            absoluteError !==
                              null
                              ? (
                                  row.forecast_error_percent ??
                                  (
                                    absoluteError /
                                    Math.abs(
                                      actualClose
                                    )
                                  ) *
                                    100
                                )
                              : null;


                          const rangeLow =
                            toFiniteOrNull(
                              row.expected_range_lower
                            );


                          const rangeHigh =
                            toFiniteOrNull(
                              row.expected_range_upper
                            );


                          return (
                            <tr
                              key={
                                `${row.symbol}-${row.base_date}-${index}`
                              }
                              className="border-b border-white/[0.035] text-[10px] last:border-0 hover:bg-white/[0.025]"
                            >

                              <td className="sticky left-0 z-10 border-r border-white/5 bg-[#0a0f17] px-4 py-3 font-semibold text-gray-300">
                                {row.base_date ||
                                  "--"}
                              </td>


                              <td className="px-3 py-3">

                                {row.history_source ===
                                "BACKFILLED_MODEL_REPLAY" ? (

                                  <span className="inline-flex rounded-full border border-blue-500/15 bg-blue-500/[0.06] px-2 py-1 text-[8px] font-semibold text-blue-400">
                                    REPLAY
                                  </span>

                                ) : (

                                  <span className="inline-flex rounded-full border border-green-500/15 bg-green-500/[0.06] px-2 py-1 text-[8px] font-semibold text-green-400">
                                    LIVE
                                  </span>

                                )}

                              </td>


                              <td className="px-3 py-3">

                                <div className="inline-flex min-w-[104px] items-center rounded-lg border border-violet-500/10 bg-violet-500/[0.045] px-3 py-2">

                                  <span className="font-bold text-violet-300">
                                    {savedForecast !==
                                    null
                                      ? formatPrice(
                                          savedForecast
                                        )
                                      : "--"}
                                  </span>

                                </div>

                              </td>


                              <td className="px-3 py-3">

                                <div className="inline-flex min-w-[104px] items-center rounded-lg border border-green-500/10 bg-green-500/[0.045] px-3 py-2">

                                  <span className="font-bold text-green-300">
                                    {resolved &&
                                    actualClose !==
                                      null
                                      ? formatPrice(
                                          actualClose
                                        )
                                      : "--"}
                                  </span>

                                </div>

                              </td>


                              <td className="px-3 py-3">

                                <div
                                  className={`inline-flex min-w-[104px] items-center rounded-lg border px-3 py-2 font-bold ${
                                    signedDifference ===
                                    null
                                      ? "border-white/5 bg-white/[0.02] text-gray-600"
                                      : Math.abs(
                                          signedDifference
                                        ) <=
                                        20
                                      ? "border-green-500/10 bg-green-500/[0.045] text-green-400"
                                      : Math.abs(
                                          signedDifference
                                        ) <=
                                        30
                                      ? "border-amber-500/10 bg-amber-500/[0.045] text-amber-400"
                                      : "border-red-500/10 bg-red-500/[0.045] text-red-400"
                                  }`}
                                >
                                  {signedDifference !==
                                  null
                                    ? `${signedDifference >=
                                      0
                                        ? "+"
                                        : "-"}${formatPrice(
                                        Math.abs(
                                          signedDifference
                                        )
                                      )}`
                                    : "--"}
                                </div>

                              </td>


                              <td className="px-3 py-3 font-semibold text-gray-300">
                                {errorPercent !==
                                null
                                  ? `${Number(
                                      errorPercent
                                    ).toFixed(
                                      2
                                    )}%`
                                  : "--"}
                              </td>


                              <td className="px-3 py-3">

                                {resolved ? (
                                  <span className="inline-flex rounded-full bg-blue-500/10 px-2.5 py-1 text-[9px] font-semibold text-blue-400">
                                    RESOLVED
                                  </span>
                                ) : (
                                  <span className="inline-flex rounded-full bg-yellow-500/10 px-2.5 py-1 text-[9px] font-semibold text-yellow-400">
                                    ⏳ PENDING
                                  </span>
                                )}

                              </td>


                              <td className="px-3 py-3 text-gray-400">
                                {row.target_date ||
                                  (
                                    resolved
                                      ? "--"
                                      : "Next trading day"
                                  )}
                              </td>


                              <td className="px-3 py-3 font-medium text-blue-300">
                                {rangeLow !==
                                  null &&
                                rangeHigh !==
                                  null
                                  ? `${formatPrice(
                                      rangeLow
                                    )} – ${formatPrice(
                                      rangeHigh
                                    )}`
                                  : "--"}
                              </td>


                              <td className="px-3 py-3">
                                {!resolved ? (
                                  <span className="text-gray-600">
                                    --
                                  </span>
                                ) : row.direction_correct ===
                                  true ? (
                                  <span className="inline-flex rounded-full bg-green-500/10 px-2 py-1 text-[9px] font-semibold text-green-400">
                                    ✓ CORRECT
                                  </span>
                                ) : row.direction_correct ===
                                  false ? (
                                  <span className="inline-flex rounded-full bg-red-500/10 px-2 py-1 text-[9px] font-semibold text-red-400">
                                    ✕ WRONG
                                  </span>
                                ) : (
                                  <span className="text-gray-600">
                                    --
                                  </span>
                                )}
                              </td>


                              <td className="px-3 py-3">
                                {!resolved ? (
                                  <span className="text-gray-600">
                                    --
                                  </span>
                                ) : row.inside_expected_range ===
                                  true ? (
                                  <span className="inline-flex rounded-full bg-green-500/10 px-2 py-1 text-[9px] font-semibold text-green-400">
                                    ✓ INSIDE
                                  </span>
                                ) : row.inside_expected_range ===
                                  false ? (
                                  <span className="inline-flex rounded-full bg-red-500/10 px-2 py-1 text-[9px] font-semibold text-red-400">
                                    ✕ OUTSIDE
                                  </span>
                                ) : (
                                  <span className="text-gray-600">
                                    --
                                  </span>
                                )}
                              </td>

                            </tr>
                          );
                        }
                      )}

                    </tbody>

                  </table>

                </div>

              </div>


              <div className="mt-3 rounded-lg border border-violet-500/10 bg-violet-500/[0.03] px-3 py-2.5">

                <p className="text-[9px] leading-4 text-gray-500">
                  <span className="font-semibold text-violet-300">
                    All-date history:
                  </span>{" "}
                  LIVE rows are genuine saved forecasts. REPLAY rows are historical model re-runs used only to fill dates that were never captured live. A 28-Aug LIVE row can correctly remain PENDING on 29-Aug because 29-Aug is Saturday; its actual value arrives on the next NSE trading day.
                </p>

              </div>

              </>

            ) : (

              <div className="px-5 py-10 text-center">

                <CheckCircle2
                  size={24}
                  className="mx-auto text-gray-700"
                />

                <p className="mt-3 text-xs font-medium text-gray-400">
                  No saved history for {historyStockInfo.short} yet
                </p>

                <p className="mt-1 text-[10px] leading-5 text-gray-600">
                  The first AI forecast you run for this stock will be saved permanently. Future trading dates will add new rows, and each row will later receive its actual close and forecast error.
                </p>

              </div>

            )}



            <div className="mt-3 rounded-xl border border-blue-500/10 bg-blue-500/[0.04] px-4 py-3">

              <p className="text-[10px] leading-5 text-gray-500">
                <span className="font-semibold text-blue-400">
                  Expected vs Actual:
                </span>
                {" "}
                X2 Point is the experimental central estimate. The 80% range is validated separately. After the next NSE trading-day close, StockVision records the X2 point error and whether the actual close landed inside the expected range.
              </p>

            </div>

          </div>


          {/* RIGHT: VALIDATION SUMMARY */}

          <div className="min-w-0 space-y-4">

            <div className="rounded-2xl border border-green-500/20 bg-[#0f141d] p-4">

              <div className="flex items-start justify-between gap-3">

                <div>

                  <h2 className="text-sm font-semibold text-white">
                    Live Prediction Validation
                  </h2>

                  <p className="mt-1 text-[10px] text-gray-600">
                    Actual model performance from saved live predictions.
                  </p>

                </div>


                <CheckCircle2
                  size={18}
                  className="text-green-400"
                />

              </div>


              <div className="mt-4 grid grid-cols-2 gap-2 border-y border-white/5 py-4 sm:grid-cols-3">

                {[
                  {
                    label: "Resolved",
                    value:
                      predictionValidation?.metrics
                        ?.resolved_predictions ??
                      0,
                    tone: "text-white",
                  },
                  {
                    label: "Pending",
                    value:
                      predictionValidation?.metrics
                        ?.pending_predictions ??
                      0,
                    tone: "text-yellow-400",
                  },
                  {
                    label: "Live Forecast MAE",
                    value:
                      predictionValidation?.metrics
                        ?.experimental_x2_point_mae !==
                        null &&
                      predictionValidation?.metrics
                        ?.experimental_x2_point_mae !==
                        undefined
                        ? formatPrice(
                            predictionValidation.metrics
                              .experimental_x2_point_mae
                          )
                        : "--",
                    tone: "text-violet-300",
                  },
                  {
                    label: "Within ₹20",
                    value:
                      predictionValidation?.metrics
                        ?.experimental_x2_within_20_percent !==
                        null &&
                      predictionValidation?.metrics
                        ?.experimental_x2_within_20_percent !==
                        undefined
                        ? `${Number(
                            predictionValidation.metrics
                              .experimental_x2_within_20_percent
                          ).toFixed(
                            1
                          )}%`
                        : "--",
                    tone: "text-green-400",
                  },
                  {
                    label: "Range Coverage",
                    value:
                      predictionValidation?.metrics
                        ?.live_range_coverage_percent !==
                        null &&
                      predictionValidation?.metrics
                        ?.live_range_coverage_percent !==
                        undefined
                        ? `${Number(
                            predictionValidation.metrics
                              .live_range_coverage_percent
                          ).toFixed(
                            1
                          )}%`
                        : "--",
                    tone: "text-blue-400",
                  },
                  {
                    label: "Over ₹30",
                    value:
                      predictionValidation?.metrics
                        ?.experimental_x2_over_30_percent !==
                        null &&
                      predictionValidation?.metrics
                        ?.experimental_x2_over_30_percent !==
                        undefined
                        ? `${Number(
                            predictionValidation.metrics
                              .experimental_x2_over_30_percent
                          ).toFixed(
                            1
                          )}%`
                        : "--",
                    tone: "text-red-400",
                  },
                ].map(
                  (
                    item
                  ) => (

                    <div
                      key={
                        item.label
                      }
                      className="min-w-0 rounded-lg bg-white/[0.025] px-2 py-3 text-center"
                    >

                      <p className="truncate text-[8px] text-gray-600">
                        {item.label}
                      </p>

                      <p className={`mt-1 text-sm font-bold ${item.tone}`}>
                        {item.value}
                      </p>

                    </div>

                  )
                )}

              </div>


              <div className="mt-4">

                <div className="mb-1.5 flex items-center justify-between">

                  <span className="text-[9px] text-gray-600">
                    Live expected-range coverage
                  </span>

                  <span className="text-[9px] font-semibold text-gray-300">
                    {predictionValidation?.metrics
                      ?.live_range_coverage_percent !==
                    null &&
                    predictionValidation?.metrics
                      ?.live_range_coverage_percent !==
                    undefined
                      ? `${Number(
                          predictionValidation.metrics
                            .live_range_coverage_percent
                        ).toFixed(
                          1
                        )}%`
                      : "--"}
                  </span>

                </div>


                <div className="h-2 overflow-hidden rounded-full bg-white/5">

                  <div
                    className="h-full rounded-full bg-gradient-to-r from-blue-500 via-violet-400 to-green-400"
                    style={{
                      width:
                        `${Math.max(
                          0,
                          Math.min(
                            100,
                            Number(
                              predictionValidation?.metrics
                                ?.live_range_coverage_percent ??
                              0
                            )
                          )
                        )}%`,
                    }}
                  />

                </div>

                <p className="mt-2 text-[9px] leading-4 text-gray-600">
                  Historical X2 range coverage was about 82% in both holdout and walk-forward evaluation. Live coverage starts after saved predictions resolve.
                </p>

              </div>

            </div>


            <div className="rounded-2xl border border-[#1b2738] bg-[#0f141d] p-4">

              <h3 className="text-sm font-semibold text-white">
                How Prediction Validation Works
              </h3>


              <div className="mt-4 space-y-3 text-[10px] leading-5 text-gray-500">

                <p>
                  <span className="font-semibold text-violet-400">
                    X2 Point:
                  </span>
                  {" "}
                  experimental next-day central estimate. Its live absolute error is tracked separately from the baseline-safe production centre.
                </p>

                <p>
                  <span className="font-semibold text-blue-400">
                    80% Range:
                  </span>
                  {" "}
                  empirical q10–q90 interval calibrated on validation data. The actual next close is marked INSIDE or OUTSIDE after resolution.
                </p>

                <p>
                  <span className="font-semibold text-green-400">
                    Within ₹20:
                  </span>
                  {" "}
                  percentage of resolved live X2 points whose absolute close error is at most ₹20.
                </p>

                <p>
                  <span className="font-semibold text-red-400">
                    Over ₹30:
                  </span>
                  {" "}
                  percentage of resolved X2 points with an absolute close error greater than ₹30.
                </p>

                <p className="border-t border-white/5 pt-3">
                  Live metrics start empty and become meaningful only as future saved predictions resolve against real NSE trading-day closes.
                </p>

              </div>

            </div>

          </div>

        </div>


        {/* =================================================
            DISCLAIMER
        ================================================= */}

        <div className="mt-4 rounded-xl border border-blue-500/10 bg-blue-500/[0.04] px-4 py-3">

          <p className="text-[10px] leading-5 text-gray-500">
            <span className="font-semibold text-blue-400">
              Disclaimer:
            </span>
            {" "}
            StockVision forecasts are probabilistic estimates based on historical and market-context data. The X2 point is experimental, the 80% range is empirically calibrated, and neither is a guaranteed future price or financial advice.
          </p>

        </div>

      </div>
    );
  }


  // =======================================================
  // FUTURE FORECAST PAGE
  // =======================================================

  function FutureForecastPage() {

    const isTraining =
      futureStatus?.status ===
      "training";


    const isError =
      futureStatus?.status ===
        "error" ||
      Boolean(
        futureError
      );


    return (
      <div>

        {/* HEADER */}

        <div className="mb-6 flex flex-col justify-between gap-4 lg:flex-row lg:items-end">

          <div>

            <div className="flex items-center gap-3">

              <div className="rounded-xl bg-blue-500/10 p-2.5">

                <TrendingUp
                  size={20}
                  className="text-blue-400"
                />

              </div>


              <div>

                <h1 className="text-2xl font-bold text-white">
                  Future Forecast
                </h1>

                <p className="mt-1 text-sm text-gray-500">

                  Multi-Horizon BiLSTM for{" "}
                  {selectedStockInfo.short}

                </p>

              </div>

            </div>

          </div>


          {futureForecast && (

            <button
              onClick={() =>
                fetchFutureForecast(
                  selectedSymbol
                )
              }
              className="rounded-xl bg-blue-500 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-blue-400"
            >
              Refresh Forecast
            </button>

          )}

        </div>


        {/* CHECKING */}

        {!futureStatus &&
        !futureForecast &&
        !futureError && (

          <LoadingBox
            text="Checking AI forecast model..."
          />

        )}


        {/* TRAINING */}

        {isTraining && (

          <TrainingProgress
            status={
              futureStatus
            }
            symbol={
              selectedSymbol
            }
          />

        )}


        {/* ERROR */}

        {isError &&
        !isTraining && (

          <div className="rounded-2xl border border-red-500/20 bg-red-500/10 p-6">

            <div className="flex items-start gap-4">

              <XCircle
                size={24}
                className="shrink-0 text-red-400"
              />


              <div>

                <p className="font-medium text-red-400">
                  Unable to prepare forecast
                </p>


                <p className="mt-2 text-sm leading-6 text-red-300/70">

                  {futureError ||
                    futureStatus?.error ||
                    "Model training failed."}

                </p>


                <button
                  onClick={() => {

                    setFutureError(
                      ""
                    );

                    setFutureStatus(
                      null
                    );

                    startFutureTraining(
                      selectedSymbol
                    );

                  }}
                  className="mt-5 rounded-xl bg-red-500/10 px-4 py-2.5 text-sm font-medium text-red-300 hover:bg-red-500/20"
                >
                  Retry Training
                </button>

              </div>

            </div>

          </div>

        )}


        {/* FORECAST LOADING */}

        {futureLoading && (

          <LoadingBox
            text="Generating 1D / 3D / 5D / 10D forecast..."
          />

        )}


        {/* FORECAST READY */}

        {futureForecast &&
        !futureLoading && (

          <>

            {/* SUMMARY CARDS */}

            <div className="mb-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">

              <StatCard
                title="Reference Close"
                value={
                  formatPrice(
                    futureForecast.current_close
                  )
                }
                subtitle={
                  futureForecast.latest_market_date
                }
              />


              <StatCard
                title="Forecast Model"
                value="BiLSTM"
                subtitle="Multi-Horizon"
              />


              <StatCard
                title="Lookback"
                value={`${futureForecast.lookback_days} Days`}
                subtitle="Historical sequence"
              />


              <StatCard
                title="Features"
                value={
                  futureForecast.features_used
                }
                subtitle="Technical + market"
              />

            </div>


            {/* CHART */}

            <div className="mb-6 rounded-xl border border-[#1b2738] bg-[#0f141d] p-5">

              <div className="mb-3 flex flex-col justify-between gap-4 lg:flex-row lg:items-center">

                <div>

                  <h2 className="font-semibold text-white">
                    Expected Market Movement
                  </h2>

                  <p className="mt-1 text-xs text-gray-500">

                    Percentage return is the primary prediction metric.

                  </p>

                </div>


                <div className="flex flex-wrap gap-4 text-[11px]">

                  <div className="flex items-center gap-2 text-green-400">

                    <span className="h-2 w-2 rounded-full bg-green-400" />

                    Positive Forecast

                  </div>


                  <div className="flex items-center gap-2 text-red-400">

                    <span className="h-2 w-2 rounded-full bg-red-400" />

                    Negative Forecast

                  </div>


                  <div className="flex items-center gap-2 text-blue-400">

                    <span className="h-2 w-2 rounded-full bg-blue-400/40" />

                    Estimated Range

                  </div>

                </div>

              </div>


              <FutureForecastChart
                forecast={
                  futureForecast
                }
              />


              <div className="border-t border-white/5 pt-4">

                <p className="text-xs leading-5 text-gray-600">

                  The shaded blue region represents validation-based
                  historical prediction error. It is an estimated range,
                  not a guaranteed confidence interval.

                </p>

              </div>

            </div>


            {/* HORIZON CARDS */}

            <div className="mb-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">

              {futureForecast.forecasts.map(
                (
                  item
                ) => (

                  <ForecastCard
                    key={
                      item.horizon
                    }
                    item={
                      item
                    }
                  />

                )
              )}

            </div>


            {/* DETAILS */}

            <div className="grid gap-6 xl:grid-cols-2">

              <div className="rounded-xl border border-[#1b2738] bg-[#0f141d] p-6">

                <h3 className="font-semibold text-white">
                  Forecast Interpretation
                </h3>


                <div className="mt-5 space-y-5">

                  <div>

                    <p className="text-sm font-medium text-white">
                      Expected Move
                    </p>

                    <p className="mt-1 text-xs leading-5 text-gray-500">

                      The predicted cumulative percentage movement from
                      the latest available daily close.

                    </p>

                  </div>


                  <div>

                    <p className="text-sm font-medium text-white">
                      Estimated Range
                    </p>

                    <p className="mt-1 text-xs leading-5 text-gray-500">

                      Generated from historical validation errors produced
                      by the trained model.

                    </p>

                  </div>


                  <div>

                    <p className="text-sm font-medium text-white">
                      Signal
                    </p>

                    <p className="mt-1 text-xs leading-5 text-gray-500">

                      Bullish only when the complete 80% estimated range
                      stays above 0%. Bearish only when the complete range
                      stays below 0%. Otherwise the result remains neutral.

                    </p>

                  </div>

                </div>

              </div>


              <div className="rounded-xl border border-[#1b2738] bg-[#0f141d] p-6">

                <h3 className="font-semibold text-white">
                  Model Evaluation
                </h3>


                <p className="mt-2 text-xs leading-5 text-gray-500">

                  Performance information is shown so forecasts are not
                  presented as guaranteed outcomes.

                </p>


                <div className="mt-5 space-y-3">

                  {futureForecast.forecasts.map(
                    (
                      item
                    ) => (

                      <div
                        key={
                          `metric-${item.horizon}`
                        }
                        className="flex items-center justify-between rounded-xl bg-white/[0.035] px-4 py-3"
                      >

                        <div>

                          <p className="text-sm font-medium text-white">
                            {item.horizon}
                          </p>


                          <p className="mt-1 text-[11px] text-gray-600">

                            Direction Accuracy:{" "}

                            {item.test_direction_accuracy_percent !==
                            null

                              ? `${Number(
                                  item.test_direction_accuracy_percent
                                ).toFixed(
                                  2
                                )}%`

                              : "--"}

                          </p>

                        </div>


                        <div className="text-right">

                          <p
                            className={`text-xs font-semibold ${
                              item.beats_naive_baseline
                                ? "text-green-400"
                                : "text-orange-400"
                            }`}
                          >

                            {item.beats_naive_baseline
                              ? "Beats Baseline"
                              : "Below Baseline"}

                          </p>


                          <p className="mt-1 text-[11px] text-gray-600">
                            {item.evaluation_status}
                          </p>

                        </div>

                      </div>

                    )
                  )}

                </div>

              </div>

            </div>


            {/* DISCLAIMER */}

            <div className="mt-6 rounded-2xl border border-yellow-500/10 bg-yellow-500/5 p-5">

              <p className="text-sm font-medium text-yellow-300">
                Forecast Notice
              </p>

              <p className="mt-2 text-sm leading-6 text-gray-500">

                StockVision forecasts possible future market movements
                using historical data and a BiLSTM model. Predictions are
                not guaranteed outcomes or investment advice.

              </p>

            </div>

          </>

        )}

      </div>
    );
  }


  // =======================================================
  // MODEL ANALYTICS PAGE
  // =======================================================

  function ModelAnalyticsPage() {

    if (
      analyticsLoading &&
      !analytics
    ) {
      return (
        <LoadingBox
          text="Loading V9 walk-forward analytics..."
        />
      );
    }


    if (
      analyticsError &&
      !analytics
    ) {
      return (
        <div className="rounded-2xl border border-red-500/20 bg-red-500/10 p-6">

          <div className="flex items-start gap-4">

            <XCircle
              size={24}
              className="shrink-0 text-red-400"
            />

            <div>

              <p className="font-medium text-red-400">
                Analytics unavailable
              </p>

              <p className="mt-2 text-sm leading-6 text-red-300/70">
                {analyticsError}
              </p>

              <button
                onClick={
                  fetchModelAnalytics
                }
                className="mt-5 rounded-xl bg-red-500/10 px-4 py-2.5 text-sm font-medium text-red-300 hover:bg-red-500/20"
              >
                Retry
              </button>

            </div>

          </div>

        </div>
      );
    }


    if (!analytics) {
      return null;
    }


    const overall =
      analytics.overall || {};


    const robustness =
      analytics.robustness || {};


    const yearly =
      analytics.yearly || [];


    const perStock =
      analytics.per_stock || [];


    const nonOverlapping =
      analytics.non_overlapping || [];


    const features =
      analytics.feature_importance || [];


    const limitations =
      analytics.limitations || [];


    const maxImportance =
      Math.max(
        ...features.map(
          (item) =>
            Number(
              item.importance || 0
            )
        ),
        0.0001
      );


    return (
      <div>

        {/* HEADER */}

        <div className="mb-6 flex flex-col justify-between gap-4 lg:flex-row lg:items-end">

          <div className="flex items-start gap-3">

            <div className="rounded-xl bg-cyan-500/10 p-2.5">

              <BarChart3
                size={21}
                className="text-cyan-400"
              />

            </div>

            <div>

              <h1 className="text-2xl font-bold text-white">
                Model Analytics
              </h1>

              <p className="mt-1 text-sm text-gray-500">
                V9 purged walk-forward evaluation and robustness analysis.
              </p>

            </div>

          </div>


          <button
            onClick={
              fetchModelAnalytics
            }
            className="rounded-xl bg-cyan-500/10 px-4 py-2.5 text-sm font-medium text-cyan-400 transition hover:bg-cyan-500/15"
          >
            Refresh Analytics
          </button>

        </div>


        {/* MODEL DESCRIPTION */}

        <div className="mb-6 rounded-2xl border border-blue-500/10 bg-blue-500/5 p-5">

          <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-center">

            <div>

              <p className="text-sm font-semibold text-white">
                {analytics.model?.name || "StockVision V9"} · {analytics.model?.engine || "Relative Strength Intelligence"}
              </p>

              <p className="mt-2 text-sm leading-6 text-gray-500">
                Target: {analytics.model?.target || "5-Day Performance vs NIFTY 50"}. Evaluation uses future-only purged walk-forward testing rather than random train/test splitting.
              </p>

            </div>

            <div className="flex flex-wrap gap-2">

              <span className="rounded-full bg-white/5 px-3 py-1.5 text-xs text-gray-300">
                {analytics.model?.stocks || 15} Stocks
              </span>

              <span className="rounded-full bg-white/5 px-3 py-1.5 text-xs text-gray-300">
                {analytics.model?.features || 29} Features
              </span>

              <span className="rounded-full bg-white/5 px-3 py-1.5 text-xs text-gray-300">
                {Number(analytics.model?.test_samples || 0).toLocaleString("en-IN")} Test Samples
              </span>

            </div>

          </div>

        </div>


        {/* OVERALL METRICS */}

        <div className="mb-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-5">

          <StatCard
            title="V9 Accuracy"
            value={`${Number(overall.model_accuracy || 0).toFixed(2)}%`}
            subtitle="3-class walk-forward"
          />

          <StatCard
            title="Macro F1"
            value={`${Number(overall.macro_f1 || 0).toFixed(2)}%`}
            subtitle="Across all classes"
          />

          <StatCard
            title="Balanced Accuracy"
            value={`${Number(overall.balanced_accuracy || 0).toFixed(2)}%`}
            subtitle="Class-balanced metric"
          />

          <StatCard
            title="vs Majority"
            value={`+${Number(overall.improvement_vs_majority || 0).toFixed(2)} pp`}
            subtitle={`${Number(overall.majority_accuracy || 0).toFixed(2)}% baseline`}
          />

          <StatCard
            title="vs Momentum"
            value={`+${Number(overall.improvement_vs_momentum || 0).toFixed(2)} pp`}
            subtitle={`${Number(overall.momentum_accuracy || 0).toFixed(2)}% baseline`}
          />

        </div>


        {/* YEARLY CHART */}

        <div className="mb-6 rounded-xl border border-[#1b2738] bg-[#0f141d] p-5">

          <div className="mb-5 flex flex-col justify-between gap-3 sm:flex-row sm:items-center">

            <div>

              <h2 className="font-semibold text-white">
                Year-by-Year Walk-Forward Performance
              </h2>

              <p className="mt-1 text-xs text-gray-500">
                Each test year is evaluated using models selected only from earlier history.
              </p>

            </div>

            <div className="flex flex-wrap gap-4 text-[11px]">

              <span className="text-blue-400">● V9 Model</span>
              <span className="text-yellow-400">● Momentum</span>
              <span className="text-gray-400">● Majority</span>

            </div>

          </div>


          <div className="h-[360px] w-full">

            <ResponsiveContainer
              width="100%"
              height="100%"
            >

              <ComposedChart
                data={yearly}
                margin={{
                  top: 20,
                  right: 25,
                  left: 0,
                  bottom: 5,
                }}
              >

                <CartesianGrid
                  stroke="#1d2531"
                  strokeDasharray="3 3"
                  vertical={false}
                />

                <XAxis
                  dataKey="year"
                  axisLine={false}
                  tickLine={false}
                  tick={{
                    fill: "#64748b",
                    fontSize: 11,
                  }}
                />

                <YAxis
                  domain={[25, 45]}
                  axisLine={false}
                  tickLine={false}
                  width={55}
                  tick={{
                    fill: "#64748b",
                    fontSize: 11,
                  }}
                  tickFormatter={
                    (value) =>
                      `${Number(value).toFixed(0)}%`
                  }
                />

                <Tooltip
                  cursor={false}
                  contentStyle={{
                    background: "#090d14",
                    border: "1px solid rgba(255,255,255,0.10)",
                    borderRadius: "12px",
                    color: "#fff",
                  }}
                  formatter={
                    (value, name) => [
                      `${Number(value).toFixed(2)}%`,
                      name === "model"
                        ? "V9 Model"
                        : name === "momentum"
                        ? "Momentum"
                        : "Majority",
                    ]
                  }
                />

                <Line
                  type="linear"
                  dataKey="model"
                  stroke="#60a5fa"
                  strokeWidth={3}
                  dot={{
                    r: 5,
                    fill: "#60a5fa",
                  }}
                  isAnimationActive={false}
                />

                <Line
                  type="linear"
                  dataKey="momentum"
                  stroke="#facc15"
                  strokeWidth={2}
                  dot={{
                    r: 4,
                    fill: "#facc15",
                  }}
                  isAnimationActive={false}
                />

                <Line
                  type="linear"
                  dataKey="majority"
                  stroke="#94a3b8"
                  strokeWidth={2}
                  strokeDasharray="6 5"
                  dot={{
                    r: 4,
                    fill: "#94a3b8",
                  }}
                  isAnimationActive={false}
                />

              </ComposedChart>

            </ResponsiveContainer>

          </div>


          <div className="mt-4 rounded-xl border border-yellow-500/10 bg-yellow-500/5 p-4">

            <p className="text-xs leading-5 text-gray-500">
              V9 does not win every year. In 2026 the majority and momentum baselines were stronger. Showing weaker periods is intentional and keeps the evaluation transparent.
            </p>

          </div>

        </div>


        {/* ROBUSTNESS SUMMARY */}

        <div className="mb-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">

          <StatCard
            title="Stocks Beating Majority"
            value={`${robustness.stocks_beating_majority || 0}/${robustness.stocks_total || 0}`}
            subtitle="Across original V9 universe"
          />

          <StatCard
            title="Stocks +3pp vs Majority"
            value={`${robustness.stocks_beating_majority_3pp || 0}/${robustness.stocks_total || 0}`}
            subtitle="Meaningful stock-level edge"
          />

          <StatCard
            title="Stocks Beating Momentum"
            value={`${robustness.stocks_beating_momentum || 0}/${robustness.stocks_total || 0}`}
            subtitle="Relative momentum baseline"
          />

          <StatCard
            title="Non-Overlap Accuracy"
            value={`${Number(robustness.non_overlap_model_accuracy || 0).toFixed(2)}%`}
            subtitle="5 independent offsets"
          />

        </div>


        {/* PER STOCK */}

        <div className="mb-6 rounded-xl border border-[#1b2738] bg-[#0f141d] p-5">

          <div className="mb-5">

            <h2 className="font-semibold text-white">
              Per-Stock Robustness
            </h2>

            <p className="mt-1 text-xs text-gray-500">
              V9 accuracy compared with stock-specific majority and relative-momentum baselines.
            </p>

          </div>


          <div className="overflow-x-auto">

            <table className="w-full min-w-[760px] text-left">

              <thead>

                <tr className="border-b border-white/5 text-xs text-gray-500">

                  <th className="px-3 py-3 font-medium">Stock</th>
                  <th className="px-3 py-3 font-medium">V9</th>
                  <th className="px-3 py-3 font-medium">Majority</th>
                  <th className="px-3 py-3 font-medium">Momentum</th>
                  <th className="px-3 py-3 font-medium">Macro F1</th>
                  <th className="px-3 py-3 font-medium">vs Majority</th>

                </tr>

              </thead>

              <tbody>

                {perStock.map(
                  (item) => {

                    const improvement =
                      Number(item.model) -
                      Number(item.majority);

                    return (
                      <tr
                        key={item.symbol}
                        className="border-b border-white/[0.035] text-sm last:border-0"
                      >

                        <td className="px-3 py-3 font-medium text-white">
                          {item.symbol}
                        </td>

                        <td className="px-3 py-3 text-blue-400">
                          {Number(item.model).toFixed(2)}%
                        </td>

                        <td className="px-3 py-3 text-gray-400">
                          {Number(item.majority).toFixed(2)}%
                        </td>

                        <td className="px-3 py-3 text-gray-400">
                          {Number(item.momentum).toFixed(2)}%
                        </td>

                        <td className="px-3 py-3 text-gray-300">
                          {Number(item.macro_f1).toFixed(2)}%
                        </td>

                        <td
                          className={`px-3 py-3 font-medium ${
                            improvement >= 0
                              ? "text-green-400"
                              : "text-red-400"
                          }`}
                        >
                          {improvement >= 0 ? "+" : ""}
                          {improvement.toFixed(2)} pp
                        </td>

                      </tr>
                    );
                  }
                )}

              </tbody>

            </table>

          </div>

        </div>


        {/* FEATURE IMPORTANCE + NON-OVERLAP */}

        <div className="mb-6 grid gap-6 xl:grid-cols-2">

          <div className="rounded-xl border border-[#1b2738] bg-[#0f141d] p-5">

            <h2 className="font-semibold text-white">
              Top V9 Features
            </h2>

            <p className="mt-1 text-xs text-gray-500">
              Diagnostic permutation importance from the V9 research pipeline.
            </p>


            <div className="mt-5 space-y-4">

              {features.slice(0, 10).map(
                (item, index) => {

                  const width =
                    Math.max(
                      4,
                      (
                        Number(item.importance || 0) /
                        maxImportance
                      ) * 100
                    );

                  return (
                    <div
                      key={item.feature}
                    >

                      <div className="mb-1.5 flex items-center justify-between gap-4">

                        <span className="text-xs text-gray-400">
                          {index + 1}. {item.feature}
                        </span>

                        <span className="text-[11px] text-gray-600">
                          {Number(item.importance).toFixed(5)}
                        </span>

                      </div>

                      <div className="h-1.5 overflow-hidden rounded-full bg-white/5">

                        <div
                          className="h-full rounded-full bg-cyan-400"
                          style={{
                            width: `${width}%`,
                          }}
                        />

                      </div>

                    </div>
                  );
                }
              )}

            </div>

          </div>


          <div className="rounded-xl border border-[#1b2738] bg-[#0f141d] p-5">

            <h2 className="font-semibold text-white">
              Non-Overlapping 5-Day Check
            </h2>

            <p className="mt-1 text-xs text-gray-500">
              Five offsets reduce the effect of overlapping 5-day targets.
            </p>


            <div className="mt-5 space-y-3">

              {nonOverlapping.map(
                (item) => (

                  <div
                    key={item.offset}
                    className="rounded-xl border border-white/5 bg-[#0b1018] p-4"
                  >

                    <div className="flex items-center justify-between gap-4">

                      <div>

                        <p className="text-sm font-medium text-white">
                          Offset {item.offset}
                        </p>

                        <p className="mt-1 text-[11px] text-gray-600">
                          {Number(item.samples).toLocaleString("en-IN")} samples
                        </p>

                      </div>

                      <p className="text-lg font-semibold text-blue-400">
                        {Number(item.model).toFixed(2)}%
                      </p>

                    </div>

                    <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-[11px] text-gray-500">

                      <span>Majority {Number(item.majority).toFixed(2)}%</span>
                      <span>Momentum {Number(item.momentum).toFixed(2)}%</span>
                      <span>Macro F1 {Number(item.macro_f1).toFixed(2)}%</span>
                      <span>Balanced {Number(item.balanced).toFixed(2)}%</span>

                    </div>

                  </div>

                )
              )}

            </div>

          </div>

        </div>


        {/* LIMITATIONS */}

        <div className="rounded-2xl border border-yellow-500/10 bg-yellow-500/5 p-6">

          <h2 className="font-semibold text-yellow-300">
            Evaluation Notes & Limitations
          </h2>

          <p className="mt-2 text-xs leading-5 text-gray-500">
            These limitations are intentionally shown because market prediction has uncertainty and model performance varies across time and stocks.
          </p>


          <div className="mt-5 grid gap-3 md:grid-cols-2">

            {limitations.map(
              (item, index) => (

                <div
                  key={`${index}-${item}`}
                  className="flex items-start gap-3 rounded-xl border border-yellow-500/10 bg-black/10 p-4"
                >

                  <CheckCircle2
                    size={16}
                    className="mt-0.5 shrink-0 text-yellow-400"
                  />

                  <p className="text-xs leading-5 text-gray-400">
                    {item}
                  </p>

                </div>

              )
            )}

          </div>

        </div>

      </div>
    );
  }


  // =======================================================
  // STOCK COMPARISON PAGE
  // =======================================================

  function StockComparisonPage() {

    const [
      comparisonSymbols,
      setComparisonSymbols,
    ] = useState([
      "RELIANCE.NS",
      "TCS.NS",
      "HDFCBANK.NS",
    ]);


    const [
      comparisonRange,
      setComparisonRange,
    ] = useState(
      "1mo"
    );


    const [
      comparisonData,
      setComparisonData,
    ] = useState(
      []
    );


    const [
      comparisonLoading,
      setComparisonLoading,
    ] = useState(
      false
    );


    const [
      comparisonError,
      setComparisonError,
    ] = useState(
      ""
    );


    const rangeLabelMap = {
      "1mo": "1 Month",
      "3mo": "3 Months",
      "6mo": "6 Months",
      "1y": "1 Year",
    };


    const shortRangeLabelMap = {
      "1mo": "1M",
      "3mo": "3M",
      "6mo": "6M",
      "1y": "1Y",
    };


    const accentClasses = [
      {
        text:
          "text-blue-400",
        bg:
          "bg-blue-500/10",
        border:
          "border-blue-500/20",
        line:
          "#3b82f6",
      },
      {
        text:
          "text-green-400",
        bg:
          "bg-green-500/10",
        border:
          "border-green-500/20",
        line:
          "#22c55e",
      },
      {
        text:
          "text-red-400",
        bg:
          "bg-red-500/10",
        border:
          "border-red-500/20",
        line:
          "#ef4444",
      },
    ];


    async function loadComparison() {

      try {

        setComparisonLoading(
          true
        );

        setComparisonError(
          ""
        );


        const results =
          await Promise.all(
            comparisonSymbols.map(
              async (
                symbol
              ) => {

                const [
                  stockResponse,
                  v9Response,
                ] =
                  await Promise.all([
                    fetch(
                      `${API_URL}/stock/${encodeURIComponent(
                        symbol
                      )}?range=${encodeURIComponent(
                        comparisonRange
                      )}`
                    ),
                    fetch(
                      `${API_URL}/relative-predict/${encodeURIComponent(
                        symbol
                      )}`
                    ),
                  ]);


                const stockData =
                  await stockResponse
                    .json()
                    .catch(
                      () =>
                        null
                    );


                const v9Data =
                  await v9Response
                    .json()
                    .catch(
                      () =>
                        null
                    );


                if (
                  !stockResponse.ok
                ) {

                  throw new Error(
                    stockData?.detail ||
                    `Unable to load ${symbol}`
                  );
                }


                const chart =
                  Array.isArray(
                    stockData?.chart
                  )
                    ? stockData.chart
                    : [];


                let periodReturn =
                  null;


                if (
                  chart.length >=
                  2
                ) {

                  const first =
                    Number(
                      chart[0]?.price
                    );


                  const last =
                    Number(
                      chart[
                        chart.length -
                        1
                      ]?.price
                    );


                  if (
                    Number.isFinite(
                      first
                    ) &&
                    Number.isFinite(
                      last
                    ) &&
                    first !== 0
                  ) {

                    periodReturn =
                      (
                        (
                          last -
                          first
                        ) /
                        first
                      ) *
                      100;
                  }
                }


                const indicators =
                  stockData?.indicators ||
                  {};


                const rsi =
                  indicators?.rsi14 ??
                  indicators?.rsi ??
                  stockData?.rsi14 ??
                  null;


                const macd =
                  typeof indicators?.macd ===
                    "object"
                    ? indicators?.macd?.macd
                    : indicators?.macd ??
                      stockData?.macd ??
                      null;


                const stockInfo =
                  allStocks.find(
                    (
                      item
                    ) =>
                      item.symbol ===
                      symbol
                  );


                return {
                  symbol,
                  short:
                    symbol.replace(
                      ".NS",
                      ""
                    ),
                  name:
                    stockInfo?.name ||
                    symbol.replace(
                      ".NS",
                      ""
                    ),
                  price:
                    Number(
                      stockData?.price
                    ),
                  changePercent:
                    Number(
                      stockData?.change_percent
                    ),
                  periodReturn,
                  volume:
                    Number(
                      stockData?.volume
                    ),
                  rsi:
                    rsi !== null &&
                    rsi !== undefined
                      ? Number(
                          rsi
                        )
                      : null,
                  macd:
                    macd !== null &&
                    macd !== undefined
                      ? Number(
                          macd
                        )
                      : null,
                  sma20:
                    Number(
                      indicators?.sma20 ??
                      indicators?.sma_20 ??
                      stockData?.sma20
                    ),
                  ema20:
                    Number(
                      indicators?.ema20 ??
                      indicators?.ema_20 ??
                      stockData?.ema20
                    ),
                  v9Signal:
                    v9Response.ok
                      ? String(
                          v9Data?.signal ||
                          "NEUTRAL"
                        ).toUpperCase()
                      : "N/A",
                  v9Score:
                    v9Response.ok
                      ? Number(
                          v9Data?.top_probability ||
                          0
                        )
                      : null,
                  chart,
                };
              }
            )
          );


        setComparisonData(
          results
        );


      } catch (
        error
      ) {

        setComparisonError(
          error.message ||
          "Unable to compare stocks."
        );


      } finally {

        setComparisonLoading(
          false
        );
      }
    }


    useEffect(
      () => {

        loadComparison();

      },
      [
        comparisonSymbols[0],
        comparisonSymbols[1],
        comparisonSymbols[2],
        comparisonRange,
      ]
    );


    function updateComparisonSymbol(
      index,
      value
    ) {

      setComparisonSymbols(
        (
          current
        ) => {

          const next = [
            ...current,
          ];


          if (
            next.includes(
              value
            )
          ) {
            return current;
          }


          next[
            index
          ] =
            value;

          return next;
        }
      );
    }


    function exportComparisonCsv() {

      if (
        comparisonData.length ===
        0
      ) {
        return;
      }


      const headers = [
        "Metric",
        ...comparisonData.map(
          (
            item
          ) =>
            item.symbol
        ),
      ];


      const rows = [
        [
          "Current Price",
          ...comparisonData.map(
            (
              item
            ) =>
              item.price
          ),
        ],
        [
          "1D Change %",
          ...comparisonData.map(
            (
              item
            ) =>
              item.changePercent
          ),
        ],
        [
          `${shortRangeLabelMap[
            comparisonRange
          ]} Return %`,
          ...comparisonData.map(
            (
              item
            ) =>
              item.periodReturn
          ),
        ],
        [
          "Volume",
          ...comparisonData.map(
            (
              item
            ) =>
              item.volume
          ),
        ],
        [
          "RSI 14",
          ...comparisonData.map(
            (
              item
            ) =>
              item.rsi
          ),
        ],
        [
          "MACD",
          ...comparisonData.map(
            (
              item
            ) =>
              item.macd
          ),
        ],
        [
          "SMA 20",
          ...comparisonData.map(
            (
              item
            ) =>
              item.sma20
          ),
        ],
        [
          "EMA 20",
          ...comparisonData.map(
            (
              item
            ) =>
              item.ema20
          ),
        ],
        [
          "V9 Signal",
          ...comparisonData.map(
            (
              item
            ) =>
              item.v9Signal
          ),
        ],
        [
          "V9 Top Raw Score",
          ...comparisonData.map(
            (
              item
            ) =>
              item.v9Score
          ),
        ],
      ];


      const csv =
        [
          headers,
          ...rows,
        ]
          .map(
            (
              row
            ) =>
              row
                .map(
                  (
                    value
                  ) =>
                    `"${String(
                      value ??
                      ""
                    ).replace(
                      /"/g,
                      '""'
                    )}"`
                )
                .join(
                  ","
                )
          )
          .join(
            "\n"
          );


      const blob =
        new Blob(
          [
            csv,
          ],
          {
            type:
              "text/csv;charset=utf-8;",
          }
        );


      const url =
        URL.createObjectURL(
          blob
        );


      const link =
        document.createElement(
          "a"
        );


      link.href =
        url;

      link.download =
        `stockvision-comparison-${comparisonRange}.csv`;

      document.body.appendChild(
        link
      );

      link.click();

      document.body.removeChild(
        link
      );

      URL.revokeObjectURL(
        url
      );
    }


    const bestPerformer =
      comparisonData.length
        ? [
            ...comparisonData,
          ].sort(
            (
              a,
              b
            ) =>
              Number(
                b.periodReturn ??
                -Infinity
              ) -
              Number(
                a.periodReturn ??
                -Infinity
              )
          )[0]
        : null;


    const biggestDecline =
      comparisonData.length
        ? [
            ...comparisonData,
          ].sort(
            (
              a,
              b
            ) =>
              Number(
                a.periodReturn ??
                Infinity
              ) -
              Number(
                b.periodReturn ??
                Infinity
              )
          )[0]
        : null;


    const strongestRsi =
      comparisonData.length
        ? [
            ...comparisonData,
          ].sort(
            (
              a,
              b
            ) =>
              Number(
                b.rsi ??
                -Infinity
              ) -
              Number(
                a.rsi ??
                -Infinity
              )
          )[0]
        : null;


    const strongestV9 =
      comparisonData.length
        ? [
            ...comparisonData,
          ].sort(
            (
              a,
              b
            ) => {

              const rank = {
                OUTPERFORM: 3,
                NEUTRAL: 2,
                UNDERPERFORM: 1,
                "N/A": 0,
              };


              const signalDifference =
                (
                  rank[
                    b.v9Signal
                  ] ||
                  0
                ) -
                (
                  rank[
                    a.v9Signal
                  ] ||
                  0
                );


              if (
                signalDifference !==
                0
              ) {
                return signalDifference;
              }


              return (
                Number(
                  b.v9Score ||
                  0
                ) -
                Number(
                  a.v9Score ||
                  0
                )
              );
            }
          )[0]
        : null;


    return (
      <div
        className="stock-comparison-reference"
        style={{
          zoom: 0.90,
          width: "111.111%",
        }}
      >

        {/* =================================================
            PAGE HEADER
        ================================================= */}

        <div className="flex flex-col justify-between gap-4 xl:flex-row xl:items-center">

          <div>

            <h1 className="text-[22px] font-bold text-white">
              Stock Comparison
            </h1>

            <p className="mt-1 text-xs text-gray-500">
              Compare multiple NSE stocks side by side with key metrics, charts and AI insights.
            </p>

          </div>


          <div className="flex flex-wrap gap-2">

            <button
              onClick={
                exportComparisonCsv
              }
              className="flex items-center gap-2 rounded-xl border border-white/10 bg-[#0f141d] px-4 py-2.5 text-xs font-medium text-gray-300 transition hover:border-white/20 hover:text-white"
            >

              <Download
                size={15}
              />

              Export CSV

            </button>


            <button
              onClick={
                loadComparison
              }
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-500 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-violet-500/15 transition hover:brightness-110"
            >

              <Activity
                size={15}
              />

              Refresh Comparison

            </button>

          </div>

        </div>


        {/* =================================================
            SELECTED STOCKS
        ================================================= */}

        <div className="mt-4 rounded-xl border border-white/5 bg-[#0f141d] p-3.5">

          <div className="grid gap-3 xl:grid-cols-[1fr_200px]">

            <div>

              <p className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-gray-500">
                Select Stocks (Max 3)
              </p>


              <div className="grid gap-2.5 md:grid-cols-3">

                {comparisonSymbols.map(
                  (
                    symbol,
                    index
                  ) => {

                    const stockInfo =
                      allStocks.find(
                        (
                          item
                        ) =>
                          item.symbol ===
                          symbol
                      );


                    const accent =
                      accentClasses[
                        index
                      ];


                    return (
                      <div
                        key={
                          `comparison-picker-${index}`
                        }
                        className="relative rounded-lg border border-white/5 bg-[#0b1018] p-2.5"
                      >

                        <div className="flex items-center gap-3">

                          <div
                            className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${accent.bg} ${accent.text}`}
                          >
                            <span className="text-xs font-bold">
                              {stockInfo?.short?.charAt(
                                0
                              ) ||
                                "S"}
                            </span>
                          </div>


                          <div className="min-w-0 flex-1">

                            <select
                              value={
                                symbol
                              }
                              onChange={
                                (
                                  event
                                ) =>
                                  updateComparisonSymbol(
                                    index,
                                    event.target.value
                                  )
                              }
                              className="w-full cursor-pointer appearance-none bg-transparent text-xs font-semibold text-white outline-none"
                            >

                              {allStocks.map(
                                (
                                  item
                                ) => (

                                  <option
                                    key={
                                      item.symbol
                                    }
                                    value={
                                      item.symbol
                                    }
                                    disabled={
                                      comparisonSymbols.includes(
                                        item.symbol
                                      ) &&
                                      item.symbol !==
                                        symbol
                                    }
                                    className="bg-[#0f141d]"
                                  >
                                    {item.symbol}
                                  </option>

                                )
                              )}

                            </select>


                            <p className="mt-1 truncate text-[10px] text-gray-600">
                              {stockInfo?.name ||
                                symbol}
                            </p>

                          </div>

                        </div>

                      </div>
                    );
                  }
                )}

              </div>

            </div>


            <div>

              <p className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-gray-500">
                Time Range
              </p>


              <div className="relative">

                <CalendarDays
                  size={14}
                  className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-gray-500"
                />


                <select
                  value={
                    comparisonRange
                  }
                  onChange={
                    (
                      event
                    ) =>
                      setComparisonRange(
                        event.target.value
                      )
                  }
                  className="w-full appearance-none rounded-lg border border-white/5 bg-[#0b1018] py-2.5 pl-9 pr-3 text-[11px] font-medium text-white outline-none focus:border-violet-500/30"
                >

                  <option
                    value="1mo"
                    className="bg-[#0f141d]"
                  >
                    1 Month
                  </option>

                  <option
                    value="3mo"
                    className="bg-[#0f141d]"
                  >
                    3 Months
                  </option>

                  <option
                    value="6mo"
                    className="bg-[#0f141d]"
                  >
                    6 Months
                  </option>

                  <option
                    value="1y"
                    className="bg-[#0f141d]"
                  >
                    1 Year
                  </option>

                </select>

              </div>

            </div>

          </div>

        </div>


        {comparisonError && (

          <div className="mt-4 rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-400">
            {comparisonError}
          </div>

        )}


        {comparisonLoading &&
        comparisonData.length ===
          0 ? (

          <div className="mt-5">

            <LoadingBox
              text="Loading live comparison..."
            />

          </div>

        ) : (

          <>

            {/* =================================================
                COMPARISON SUMMARY
            ================================================= */}

            <div className="mt-3 overflow-hidden rounded-xl border border-white/5 bg-[#0f141d]">

              <div className="border-b border-white/5 px-4 py-2.5">

                <h2 className="text-sm font-semibold text-white">
                  Comparison Summary (Live)
                </h2>

              </div>


              <div className="overflow-x-auto">

                <table className="w-full min-w-[820px] table-fixed">

                  <thead>

                    <tr className="border-b border-white/5 text-left text-[10px] text-gray-500">

                      <th className="w-[20%] px-4 py-2.5 font-medium">
                        Metric
                      </th>


                      {comparisonData.map(
                        (
                          item,
                          index
                        ) => (

                          <th
                            key={
                              `comparison-header-${item.symbol}`
                            }
                            className={`px-4 py-2.5 font-semibold ${accentClasses[index]?.text}`}
                          >
                            {item.symbol}
                          </th>

                        )
                      )}

                    </tr>

                  </thead>


                  <tbody>

                    {[
                      {
                        label:
                          "Current Price",
                        get:
                          (
                            item
                          ) =>
                            formatPrice(
                              item.price
                            ),
                      },
                      {
                        label:
                          "1D Change",
                        get:
                          (
                            item
                          ) =>
                            formatPercent(
                              item.changePercent
                            ),
                        color:
                          (
                            item
                          ) =>
                            Number(
                              item.changePercent
                            ) >= 0
                              ? "text-green-400"
                              : "text-red-400",
                      },
                      {
                        label:
                          `${shortRangeLabelMap[
                            comparisonRange
                          ]} Return`,
                        get:
                          (
                            item
                          ) =>
                            formatPercent(
                              item.periodReturn
                            ),
                        color:
                          (
                            item
                          ) =>
                            Number(
                              item.periodReturn
                            ) >= 0
                              ? "text-green-400"
                              : "text-red-400",
                      },
                      {
                        label:
                          "Volume",
                        get:
                          (
                            item
                          ) =>
                            formatVolume(
                              item.volume
                            ),
                      },
                      {
                        label:
                          "RSI (14)",
                        get:
                          (
                            item
                          ) =>
                            item.rsi !==
                              null
                              ? item.rsi.toFixed(
                                  2
                                )
                              : "--",
                        color:
                          (
                            item
                          ) =>
                            item.rsi !==
                              null &&
                            item.rsi >=
                              50
                              ? "text-green-400"
                              : "text-yellow-400",
                      },
                      {
                        label:
                          "MACD",
                        get:
                          (
                            item
                          ) =>
                            item.macd !==
                              null
                              ? item.macd.toFixed(
                                  2
                                )
                              : "--",
                        color:
                          (
                            item
                          ) =>
                            item.macd !==
                              null &&
                            item.macd >=
                              0
                              ? "text-green-400"
                              : "text-red-400",
                      },
                      {
                        label:
                          "SMA 20",
                        get:
                          (
                            item
                          ) =>
                            Number.isFinite(
                              item.sma20
                            )
                              ? formatPrice(
                                  item.sma20
                                )
                              : "--",
                      },
                      {
                        label:
                          "EMA 20",
                        get:
                          (
                            item
                          ) =>
                            Number.isFinite(
                              item.ema20
                            )
                              ? formatPrice(
                                  item.ema20
                                )
                              : "--",
                      },
                      {
                        label:
                          "V9 Signal (5D)",
                        signal:
                          true,
                      },
                      {
                        label:
                          "V9 Top Raw Score",
                        get:
                          (
                            item
                          ) =>
                            item.v9Score !==
                              null
                              ? formatProbabilityScore(
                                  item.v9Score
                                )
                              : "--",
                      },
                    ].map(
                      (
                        row
                      ) => (

                        <tr
                          key={
                            row.label
                          }
                          className="border-b border-white/[0.035] last:border-0"
                        >

                          <td className="px-4 py-1.5 text-[10px] font-medium text-gray-300">
                            {row.label}
                          </td>


                          {comparisonData.map(
                            (
                              item
                            ) => (

                              <td
                                key={
                                  `${row.label}-${item.symbol}`
                                }
                                className={`px-4 py-1.5 text-[10px] font-medium ${
                                  row.color
                                    ? row.color(
                                        item
                                      )
                                    : "text-gray-300"
                                }`}
                              >

                                {row.signal ? (

                                  <span
                                    className={`inline-flex rounded-md px-2 py-1 text-[9px] font-semibold ${
                                      item.v9Signal ===
                                      "OUTPERFORM"
                                        ? "bg-green-500/10 text-green-400"
                                        : item.v9Signal ===
                                          "UNDERPERFORM"
                                        ? "bg-red-500/10 text-red-400"
                                        : "bg-yellow-500/10 text-yellow-400"
                                    }`}
                                  >
                                    {item.v9Signal}
                                  </span>

                                ) : (

                                  row.get(
                                    item
                                  )

                                )}

                              </td>

                            )
                          )}

                        </tr>

                      )
                    )}

                  </tbody>

                </table>

              </div>

            </div>


            {/* =================================================
                PRICE TREND CHARTS
            ================================================= */}

            <div className="mt-4 flex items-center justify-between gap-4">

              <h2 className="text-sm font-semibold text-white">
                {rangeLabelMap[
                  comparisonRange
                ]} Price Trend
              </h2>

              <span className="text-[10px] text-gray-600">
                Live historical market data
              </span>

            </div>


            <div className="mt-2.5 grid gap-3 xl:grid-cols-3">

              {comparisonData.map(
                (
                  item,
                  index
                ) => {

                  const accent =
                    accentClasses[
                      index
                    ];


                  return (
                    <div
                      key={
                        `comparison-trend-${item.symbol}`
                      }
                      className="rounded-xl border border-white/5 bg-[#0f141d] p-3.5"
                    >

                      <div className="flex items-center justify-between">

                        <p
                          className={`text-sm font-semibold ${accent.text}`}
                        >
                          {item.symbol}
                        </p>


                        <p
                          className={`text-sm font-semibold ${
                            Number(
                              item.periodReturn
                            ) >= 0
                              ? "text-green-400"
                              : "text-red-400"
                          }`}
                        >
                          {formatPercent(
                            item.periodReturn
                          )}
                        </p>

                      </div>


                      <div className="mt-2.5 h-[145px]">

                        <ResponsiveContainer
                          width="100%"
                          height="100%"
                        >

                          <ComposedChart
                            data={
                              item.chart
                            }
                            margin={{
                              top: 5,
                              right: 2,
                              left: 0,
                              bottom: 0,
                            }}
                          >

                            <defs>

                              <linearGradient
                                id={`comparison-fill-${index}`}
                                x1="0"
                                y1="0"
                                x2="0"
                                y2="1"
                              >

                                <stop
                                  offset="0%"
                                  stopColor={
                                    accent.line
                                  }
                                  stopOpacity={0.20}
                                />

                                <stop
                                  offset="100%"
                                  stopColor={
                                    accent.line
                                  }
                                  stopOpacity={0}
                                />

                              </linearGradient>

                            </defs>


                            <CartesianGrid
                              stroke="#1d2531"
                              strokeDasharray="3 3"
                              vertical={false}
                            />


                            <XAxis
                              dataKey="time"
                              axisLine={false}
                              tickLine={false}
                              minTickGap={35}
                              tick={{
                                fill:
                                  "#64748b",
                                fontSize:
                                  9,
                              }}
                            />


                            <YAxis
                              orientation="right"
                              axisLine={false}
                              tickLine={false}
                              width={42}
                              domain={[
                                "auto",
                                "auto",
                              ]}
                              tick={{
                                fill:
                                  "#64748b",
                                fontSize:
                                  9,
                              }}
                              tickFormatter={
                                (
                                  value
                                ) =>
                                  Number(
                                    value
                                  ).toFixed(
                                    0
                                  )
                              }
                            />


                            <Tooltip
                              content={
                                <MarketChartTooltip />
                              }
                            />


                            <Area
                              type="monotone"
                              dataKey="price"
                              stroke="none"
                              fill={`url(#comparison-fill-${index})`}
                              isAnimationActive={
                                false
                              }
                            />


                            <Line
                              type="monotone"
                              dataKey="price"
                              stroke={
                                accent.line
                              }
                              strokeWidth={2}
                              dot={false}
                              isAnimationActive={
                                false
                              }
                            />

                          </ComposedChart>

                        </ResponsiveContainer>

                      </div>

                    </div>
                  );
                }
              )}

            </div>


            {/* =================================================
                KEY INSIGHTS
            ================================================= */}

            <h2 className="mt-4 text-sm font-semibold text-white">
              Key Insights
            </h2>


            <div className="mt-2.5 grid gap-2.5 md:grid-cols-2 xl:grid-cols-4">

              <div className="rounded-xl border border-green-500/20 bg-[linear-gradient(145deg,rgba(34,197,94,0.09),rgba(15,20,29,0.92))] p-3.5">

                <div className="flex items-center gap-3">

                  <div className="rounded-xl bg-green-500/10 p-2.5">

                    <TrendingUp
                      size={18}
                      className="text-green-400"
                    />

                  </div>


                  <div>

                    <p className="text-[10px] text-gray-400">
                      Best {shortRangeLabelMap[
                        comparisonRange
                      ]} Performer
                    </p>

                    <p className="mt-1 text-lg font-bold text-green-400">
                      {bestPerformer?.symbol ||
                        "--"}
                    </p>

                  </div>

                </div>


                <p className="mt-3 text-xl font-bold text-green-400">
                  {bestPerformer
                    ? formatPercent(
                        bestPerformer.periodReturn
                      )
                    : "--"}
                </p>


                <p className="mt-2 text-[10px] text-gray-500">
                  Best return among selected stocks
                </p>

              </div>


              <div className="rounded-xl border border-red-500/20 bg-[linear-gradient(145deg,rgba(239,68,68,0.08),rgba(15,20,29,0.92))] p-3.5">

                <div className="flex items-center gap-3">

                  <div className="rounded-xl bg-red-500/10 p-2.5">

                    <TrendingDown
                      size={18}
                      className="text-red-400"
                    />

                  </div>


                  <div>

                    <p className="text-[10px] text-gray-400">
                      Biggest {shortRangeLabelMap[
                        comparisonRange
                      ]} Decline
                    </p>

                    <p className="mt-1 text-lg font-bold text-red-400">
                      {biggestDecline?.symbol ||
                        "--"}
                    </p>

                  </div>

                </div>


                <p className="mt-3 text-xl font-bold text-red-400">
                  {biggestDecline
                    ? formatPercent(
                        biggestDecline.periodReturn
                      )
                    : "--"}
                </p>


                <p className="mt-2 text-[10px] text-gray-500">
                  Weakest return among selected stocks
                </p>

              </div>


              <div className="rounded-xl border border-green-500/20 bg-[linear-gradient(145deg,rgba(16,185,129,0.08),rgba(15,20,29,0.92))] p-3.5">

                <div className="flex items-center gap-3">

                  <div className="rounded-xl bg-emerald-500/10 p-2.5">

                    <Gauge
                      size={18}
                      className="text-emerald-400"
                    />

                  </div>


                  <div>

                    <p className="text-[10px] text-gray-400">
                      Highest RSI
                    </p>

                    <p className="mt-1 text-lg font-bold text-emerald-400">
                      {strongestRsi?.symbol ||
                        "--"}
                    </p>

                  </div>

                </div>


                <p className="mt-3 text-xl font-bold text-emerald-400">

                  {strongestRsi?.rsi !==
                    null &&
                  strongestRsi?.rsi !==
                    undefined
                    ? strongestRsi.rsi.toFixed(
                        2
                      )
                    : "--"}

                </p>


                <p className="mt-2 text-[10px] text-gray-500">
                  Strongest current RSI among selected stocks
                </p>

              </div>


              <div className="rounded-xl border border-yellow-500/20 bg-[linear-gradient(145deg,rgba(234,179,8,0.08),rgba(15,20,29,0.92))] p-3.5">

                <div className="flex items-center gap-3">

                  <div className="rounded-xl bg-yellow-500/10 p-2.5">

                    <Star
                      size={18}
                      className="text-yellow-400"
                    />

                  </div>


                  <div>

                    <p className="text-[10px] text-gray-400">
                      V9 Top Signal
                    </p>

                    <p className="mt-1 text-lg font-bold text-yellow-400">
                      {strongestV9?.symbol ||
                        "--"}
                    </p>

                  </div>

                </div>


                <div className="mt-3">

                  <RelativeStrengthBadge
                    signal={
                      strongestV9?.v9Signal ||
                      "NEUTRAL"
                    }
                  />

                </div>


                <p className="mt-3 text-[10px] text-gray-500">
                  Strongest V9 relative-strength result
                </p>

              </div>

            </div>


            <div className="mt-3 rounded-lg border border-yellow-500/10 bg-yellow-500/[0.03] px-4 py-3">

              <p className="text-[10px] leading-4 text-gray-600">
                Comparison metrics use live / latest available market data. V9 values are raw model outputs and should not be interpreted as guaranteed investment outcomes.
              </p>

            </div>

          </>

        )}

      </div>
    );
  }


  // =======================================================
  // ALERTS
  // =======================================================

  function AlertsPage() {

    const ALERTS_KEY =
      "stockvision_alerts";


    const [
      alerts,
      setAlerts,
    ] = useState(
      () => {

        try {

          const saved =
            localStorage.getItem(
              ALERTS_KEY
            );


          return saved
            ? JSON.parse(
                saved
              )
            : [];

        } catch {

          return [];
        }
      }
    );


    const [
      alertSymbol,
      setAlertSymbol,
    ] = useState(
      selectedSymbol
    );


    const [
      alertType,
      setAlertType,
    ] = useState(
      "PRICE_ABOVE"
    );


    const [
      alertTarget,
      setAlertTarget,
    ] = useState(
      ""
    );


    const [
      alertChecking,
      setAlertChecking,
    ] = useState(
      false
    );


    const [
      alertMessage,
      setAlertMessage,
    ] = useState(
      ""
    );


    const [
      alertError,
      setAlertError,
    ] = useState(
      ""
    );


    const priceAlert =
      alertType ===
      "PRICE_ABOVE" ||
      alertType ===
      "PRICE_BELOW";


    useEffect(
      () => {

        try {

          localStorage.setItem(
            ALERTS_KEY,
            JSON.stringify(
              alerts
            )
          );

        } catch {

          // Ignore localStorage failures.
        }

      },
      [
        alerts,
      ]
    );


    useEffect(
      () => {

        setAlertSymbol(
          selectedSymbol
        );

      },
      [
        selectedSymbol,
      ]
    );


    function alertTypeLabel(
      type
    ) {

      if (
        type ===
        "PRICE_ABOVE"
      ) {

        return "Price Above";
      }


      if (
        type ===
        "PRICE_BELOW"
      ) {

        return "Price Below";
      }


      if (
        type ===
        "V9_OUTPERFORM"
      ) {

        return "V9 → OUTPERFORM";
      }


      return "V9 → UNDERPERFORM";
    }


    function alertStockLabel(
      symbol
    ) {

      const info =
        allStocks.find(
          (
            item
          ) =>
            item.symbol ===
            symbol
        );


      return info?.short ||
        symbol.replace(
          ".NS",
          ""
        );
    }


    function conditionText(
      item
    ) {

      if (
        item.type ===
        "PRICE_ABOVE"
      ) {

        return `Price ≥ ${formatPrice(
          Number(
            item.target
          )
        )}`;
      }


      if (
        item.type ===
        "PRICE_BELOW"
      ) {

        return `Price ≤ ${formatPrice(
          Number(
            item.target
          )
        )}`;
      }


      if (
        item.type ===
        "V9_OUTPERFORM"
      ) {

        return "V9 signal becomes OUTPERFORM";
      }


      return "V9 signal becomes UNDERPERFORM";
    }


    async function enableBrowserNotifications() {

      setAlertMessage(
        ""
      );


      setAlertError(
        ""
      );


      if (
        !(
          "Notification" in window
        )
      ) {

        setAlertError(
          "Browser notifications are not supported here."
        );

        return;
      }


      try {

        const permission =
          await Notification.requestPermission();


        if (
          permission ===
          "granted"
        ) {

          setAlertMessage(
            "Browser notifications enabled."
          );

        } else {

          setAlertError(
            "Browser notification permission was not granted."
          );
        }

      } catch {

        setAlertError(
          "Could not request browser notification permission."
        );
      }
    }


    function createAlert() {

      setAlertMessage(
        ""
      );


      setAlertError(
        ""
      );


      if (
        !alertSymbol
      ) {

        setAlertError(
          "Choose a stock first."
        );

        return;
      }


      if (
        priceAlert
      ) {

        const target =
          Number(
            alertTarget
          );


        if (
          !Number.isFinite(
            target
          ) ||
          target <=
          0
        ) {

          setAlertError(
            "Enter a valid target price."
          );

          return;
        }
      }


      const duplicate =
        alerts.some(
          (
            item
          ) =>

            item.symbol ===
            alertSymbol

            &&

            item.type ===
            alertType

            &&

            (
              !priceAlert

              ||

              Number(
                item.target
              ) ===
                Number(
                  alertTarget
                )
            )
        );


      if (
        duplicate
      ) {

        setAlertError(
          "This alert already exists."
        );

        return;
      }


      const newAlert = {
        id:
          `${Date.now()}-${Math.random()
            .toString(
              36
            )
            .slice(
              2,
              8
            )}`,
        symbol:
          alertSymbol,
        type:
          alertType,
        target:
          priceAlert
            ? Number(
                alertTarget
              )
            : null,
        enabled:
          true,
        triggered:
          false,
        createdAt:
          new Date()
            .toISOString(),
        lastCheckedAt:
          null,
        lastValue:
          null,
        triggeredAt:
          null,
      };


      setAlerts(
        (
          current
        ) => [
          newAlert,
          ...current,
        ]
      );


      setAlertTarget(
        ""
      );


      setAlertMessage(
        "Alert created successfully."
      );
    }


    function removeAlert(
      id
    ) {

      setAlerts(
        (
          current
        ) =>
          current.filter(
            (
              item
            ) =>
              item.id !==
              id
          )
      );
    }


    function toggleAlert(
      id
    ) {

      setAlerts(
        (
          current
        ) =>
          current.map(
            (
              item
            ) =>
              item.id ===
              id
                ? {
                    ...item,
                    enabled:
                      !item.enabled,
                    triggered:
                      false,
                    triggeredAt:
                      null,
                  }
                : item
          )
      );
    }


    async function evaluateAlert(
      item
    ) {

      if (
        !item.enabled
      ) {

        return item;
      }


      const now =
        new Date()
          .toISOString();


      try {

        let currentValue =
          null;


        let triggered =
          false;


        if (
          item.type ===
          "PRICE_ABOVE" ||
          item.type ===
          "PRICE_BELOW"
        ) {

          const response =
            await fetch(
              `${API_URL}/stock/${encodeURIComponent(
                item.symbol
              )}?range=1d`
            );


          const data =
            await response
              .json()
              .catch(
                () =>
                  null
              );


          if (
            !response.ok
          ) {

            throw new Error(
              data?.detail ||
              "Price unavailable"
            );
          }


          currentValue =
            Number(
              data?.price
            );


          if (
            !Number.isFinite(
              currentValue
            )
          ) {

            throw new Error(
              "Invalid price"
            );
          }


          triggered =
            item.type ===
            "PRICE_ABOVE"
              ? currentValue >=
                Number(
                  item.target
                )
              : currentValue <=
                Number(
                  item.target
                );

        } else {

          const response =
            await fetch(
              `${API_URL}/relative-predict/${encodeURIComponent(
                item.symbol
              )}`
            );


          const data =
            await response
              .json()
              .catch(
                () =>
                  null
              );


          if (
            !response.ok
          ) {

            throw new Error(
              data?.detail ||
              "V9 signal unavailable"
            );
          }


          currentValue =
            String(
              data?.signal ||
              "NEUTRAL"
            ).toUpperCase();


          triggered =
            item.type ===
            "V9_OUTPERFORM"
              ? currentValue ===
                "OUTPERFORM"
              : currentValue ===
                "UNDERPERFORM";
        }


        const newlyTriggered =
          triggered &&
          !item.triggered;


        if (
          newlyTriggered &&
          notificationsEnabled &&
          "Notification" in
            window &&
          Notification.permission ===
            "granted"
        ) {

          try {

            new Notification(
              `StockVision Alert — ${alertStockLabel(
                item.symbol
              )}`,
              {
                body:
                  item.type ===
                    "PRICE_ABOVE" ||
                  item.type ===
                    "PRICE_BELOW"
                    ? `${conditionText(
                        item
                      )}. Current: ${formatPrice(
                        Number(
                          currentValue
                        )
                      )}`
                    : `${conditionText(
                        item
                      )}. Current signal: ${currentValue}`,
              }
            );

          } catch {

            // Browser notification is optional.
          }
        }


        return {
          ...item,
          triggered,
          lastCheckedAt:
            now,
          lastValue:
            currentValue,
          triggeredAt:
            newlyTriggered
              ? now
              : item.triggeredAt,
          checkError:
            null,
        };

      } catch (
        error
      ) {

        return {
          ...item,
          lastCheckedAt:
            now,
          checkError:
            error.message ||
            "Check failed",
        };
      }
    }


    async function checkAlerts() {

      if (
        alertChecking
      ) {

        return;
      }


      setAlertChecking(
        true
      );


      setAlertError(
        ""
      );


      try {

        const checked =
          await Promise.all(
            alerts.map(
              (
                item
              ) =>
                evaluateAlert(
                  item
                )
            )
          );


        setAlerts(
          checked
        );


        setAlertMessage(
          checked.length
            ? "Alerts refreshed with current market data."
            : "No alerts to check."
        );

      } catch (
        error
      ) {

        setAlertError(
          error.message ||
          "Unable to refresh alerts."
        );

      } finally {

        setAlertChecking(
          false
        );
      }
    }


    useEffect(
      () => {

        if (
          alerts.length ===
          0
        ) {

          return undefined;
        }


        const timer =
          window.setInterval(
            () => {

              checkAlerts();

            },
            60000
          );


        return () =>
          window.clearInterval(
            timer
          );

      },
      [
        alerts.length,
      ]
    );


    const activeCount =
      alerts.filter(
        (
          item
        ) =>
          item.enabled &&
          !item.triggered
      ).length;


    const triggeredCount =
      alerts.filter(
        (
          item
        ) =>
          item.triggered
      ).length;


    return (
      <div>

        <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-end">

          <div>

            <h1 className="text-2xl font-bold text-white">
              Alerts
            </h1>

            <p className="mt-1 text-sm text-gray-500">
              Create live price and V9 model-signal alerts for NSE stocks.
            </p>

          </div>


          <div className="flex flex-wrap gap-2">

            <button
              onClick={
                enableBrowserNotifications
              }
              className="rounded-lg border border-white/10 bg-[#0f141d] px-3 py-2 text-[11px] font-medium text-gray-300 transition hover:border-violet-500/30 hover:text-white"
            >
              Enable Browser Notifications
            </button>


            <button
              onClick={
                checkAlerts
              }
              disabled={
                alertChecking
              }
              className="flex items-center gap-2 rounded-lg bg-violet-500 px-3 py-2 text-[11px] font-semibold text-white transition hover:bg-violet-400 disabled:cursor-wait disabled:opacity-60"
            >

              {alertChecking ? (

                <LoaderCircle
                  size={14}
                  className="animate-spin"
                />

              ) : (

                <Bell
                  size={14}
                />

              )}

              {alertChecking
                ? "Checking..."
                : "Check Alerts"}

            </button>

          </div>

        </div>


        <div className="mt-5 grid gap-3 md:grid-cols-3">

          <div className="rounded-xl border border-white/5 bg-[#0f141d] p-4">

            <p className="text-[10px] uppercase tracking-wider text-gray-600">
              Total Alerts
            </p>

            <p className="mt-2 text-2xl font-bold text-white">
              {alerts.length}
            </p>

          </div>


          <div className="rounded-xl border border-blue-500/10 bg-[#0f141d] p-4">

            <p className="text-[10px] uppercase tracking-wider text-gray-600">
              Watching
            </p>

            <p className="mt-2 text-2xl font-bold text-blue-400">
              {activeCount}
            </p>

          </div>


          <div className="rounded-xl border border-green-500/10 bg-[#0f141d] p-4">

            <p className="text-[10px] uppercase tracking-wider text-gray-600">
              Triggered
            </p>

            <p className="mt-2 text-2xl font-bold text-green-400">
              {triggeredCount}
            </p>

          </div>

        </div>


        <div className="mt-4 grid gap-4 xl:grid-cols-[0.9fr_1.4fr]">

          <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-5">

            <div className="flex items-center gap-2">

              <Plus
                size={16}
                className="text-violet-400"
              />

              <h2 className="text-sm font-semibold text-white">
                Create Alert
              </h2>

            </div>


            <div className="mt-5 space-y-4">

              <div>

                <label className="text-[10px] uppercase tracking-wider text-gray-600">
                  Stock
                </label>

                <select
                  value={
                    alertSymbol
                  }
                  onChange={
                    (
                      event
                    ) =>
                      setAlertSymbol(
                        event.target.value
                      )
                  }
                  className="mt-2 w-full rounded-xl border border-white/5 bg-[#0b1018] px-3 py-3 text-sm text-white outline-none focus:border-violet-500/30"
                >

                  {allStocks.map(
                    (
                      item
                    ) => (

                      <option
                        key={
                          item.symbol
                        }
                        value={
                          item.symbol
                        }
                      >
                        {item.short} — {item.name}
                      </option>

                    )
                  )}

                </select>

              </div>


              <div>

                <label className="text-[10px] uppercase tracking-wider text-gray-600">
                  Alert Type
                </label>

                <select
                  value={
                    alertType
                  }
                  onChange={
                    (
                      event
                    ) => {

                      setAlertType(
                        event.target.value
                      );

                      setAlertTarget(
                        ""
                      );
                    }
                  }
                  className="mt-2 w-full rounded-xl border border-white/5 bg-[#0b1018] px-3 py-3 text-sm text-white outline-none focus:border-violet-500/30"
                >

                  <option value="PRICE_ABOVE">
                    Price Above
                  </option>

                  <option value="PRICE_BELOW">
                    Price Below
                  </option>

                  <option value="V9_OUTPERFORM">
                    V9 → OUTPERFORM
                  </option>

                  <option value="V9_UNDERPERFORM">
                    V9 → UNDERPERFORM
                  </option>

                </select>

              </div>


              {priceAlert && (

                <div>

                  <label className="text-[10px] uppercase tracking-wider text-gray-600">
                    Target Price
                  </label>

                  <input
                    type="number"
                    min="0"
                    step="0.05"
                    value={
                      alertTarget
                    }
                    onChange={
                      (
                        event
                      ) =>
                        setAlertTarget(
                          event.target.value
                        )
                    }
                    placeholder="e.g. 1300"
                    className="mt-2 w-full rounded-xl border border-white/5 bg-[#0b1018] px-3 py-3 text-sm text-white outline-none placeholder:text-gray-700 focus:border-violet-500/30"
                  />

                </div>

              )}


              <button
                onClick={
                  createAlert
                }
                className="w-full rounded-xl bg-gradient-to-r from-violet-500 to-purple-500 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-500/10 transition hover:from-violet-400 hover:to-purple-400"
              >
                Create Alert
              </button>


              {alertMessage && (

                <div className="rounded-lg border border-green-500/10 bg-green-500/[0.04] px-3 py-2.5 text-[11px] text-green-400">
                  {alertMessage}
                </div>

              )}


              {alertError && (

                <div className="rounded-lg border border-red-500/10 bg-red-500/[0.04] px-3 py-2.5 text-[11px] text-red-400">
                  {alertError}
                </div>

              )}

            </div>

          </div>


          <div className="min-w-0 rounded-2xl border border-white/5 bg-[#0f141d] p-5">

            <div className="flex items-center justify-between gap-3">

              <div>

                <h2 className="text-sm font-semibold text-white">
                  Saved Alerts
                </h2>

                <p className="mt-1 text-[10px] text-gray-600">
                  Alerts are stored locally and checked with live StockVision API data.
                </p>

              </div>

              <span className="rounded-full border border-white/5 bg-[#0b1018] px-2.5 py-1 text-[10px] text-gray-500">
                {alerts.length} total
              </span>

            </div>


            {alerts.length ===
              0 ? (

              <div className="mt-5 rounded-xl border border-dashed border-white/10 bg-[#0b1018]/60 px-4 py-10 text-center">

                <Bell
                  size={25}
                  className="mx-auto text-gray-700"
                />

                <p className="mt-3 text-sm font-medium text-gray-400">
                  No alerts yet
                </p>

                <p className="mt-1 text-[11px] text-gray-600">
                  Create your first price or V9 signal alert.
                </p>

              </div>

            ) : (

              <div className="mt-4 space-y-3">

                {alerts.map(
                  (
                    item
                  ) => {

                    const priceType =
                      item.type ===
                        "PRICE_ABOVE" ||
                      item.type ===
                        "PRICE_BELOW";


                    return (
                      <div
                        key={
                          item.id
                        }
                        className={`rounded-xl border p-4 ${
                          item.triggered
                            ? "border-green-500/20 bg-green-500/[0.04]"
                            : item.enabled
                            ? "border-white/5 bg-[#0b1018]"
                            : "border-white/5 bg-[#0b1018]/50 opacity-60"
                        }`}
                      >

                        <div className="flex flex-col justify-between gap-3 md:flex-row md:items-start">

                          <div className="min-w-0">

                            <div className="flex flex-wrap items-center gap-2">

                              <p className="text-sm font-semibold text-white">
                                {alertStockLabel(
                                  item.symbol
                                )}
                              </p>

                              <span className="rounded-full border border-violet-500/20 bg-violet-500/10 px-2 py-0.5 text-[9px] font-medium text-violet-300">
                                {alertTypeLabel(
                                  item.type
                                )}
                              </span>

                              {item.triggered && (

                                <span className="rounded-full border border-green-500/20 bg-green-500/10 px-2 py-0.5 text-[9px] font-semibold text-green-400">
                                  TRIGGERED
                                </span>

                              )}

                            </div>

                            <p className="mt-2 text-[12px] text-gray-300">
                              {conditionText(
                                item
                              )}
                            </p>

                            <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-[10px] text-gray-600">

                              <span>
                                Current:{" "}
                                <strong className="font-medium text-gray-400">
                                  {item.lastValue ===
                                  null ||
                                  item.lastValue ===
                                  undefined
                                    ? "--"
                                    : priceType
                                    ? formatPrice(
                                        Number(
                                          item.lastValue
                                        )
                                      )
                                    : String(
                                        item.lastValue
                                      )}
                                </strong>
                              </span>

                              <span>
                                Last checked:{" "}
                                <strong className="font-medium text-gray-400">
                                  {item.lastCheckedAt
                                    ? new Date(
                                        item.lastCheckedAt
                                      ).toLocaleTimeString(
                                        [],
                                        {
                                          hour:
                                            "2-digit",
                                          minute:
                                            "2-digit",
                                        }
                                      )
                                    : "--"}
                                </strong>
                              </span>

                            </div>

                            {item.checkError && (

                              <p className="mt-2 text-[10px] text-red-400">
                                {item.checkError}
                              </p>

                            )}

                          </div>


                          <div className="flex shrink-0 gap-2">

                            <button
                              onClick={() =>
                                toggleAlert(
                                  item.id
                                )
                              }
                              className={`rounded-lg border px-3 py-2 text-[10px] font-medium transition ${
                                item.enabled
                                  ? "border-blue-500/20 bg-blue-500/10 text-blue-400 hover:bg-blue-500/15"
                                  : "border-white/10 bg-white/[0.03] text-gray-500 hover:text-white"
                              }`}
                            >
                              {item.enabled
                                ? "Enabled"
                                : "Disabled"}
                            </button>

                            <button
                              onClick={() =>
                                removeAlert(
                                  item.id
                                )
                              }
                              className="rounded-lg border border-red-500/10 bg-red-500/[0.04] px-3 py-2 text-[10px] font-medium text-red-400 transition hover:bg-red-500/10"
                            >
                              Remove
                            </button>

                          </div>

                        </div>

                      </div>
                    );
                  }
                )}

              </div>

            )}

          </div>

        </div>


        <div className="mt-4 rounded-xl border border-blue-500/10 bg-blue-500/[0.03] px-4 py-3">

          <p className="text-[10px] leading-5 text-gray-500">
            Alerts are checked while StockVision is open. Price alerts use live/latest StockVision market data. V9 alerts use the relative-strength endpoint. Browser notifications require permission; in-app alert status still works without it.
          </p>

        </div>

      </div>
    );
  }


  // =======================================================
  // NEWS & SENTIMENT
  // =======================================================

  function NewsSentimentPage() {

    const [
      newsData,
      setNewsData,
    ] = useState(
      null
    );


    const [
      newsLoading,
      setNewsLoading,
    ] = useState(
      false
    );


    const [
      newsError,
      setNewsError,
    ] = useState(
      ""
    );


    const [
      newsPage,
      setNewsPage,
    ] = useState(
      1
    );


    const PAGE_SIZE =
      5;


    async function loadNewsSentiment() {

      setNewsLoading(
        true
      );


      setNewsError(
        ""
      );


      try {

        const response =
          await fetch(
            `${API_URL}/news-sentiment/${encodeURIComponent(
              selectedSymbol
            )}`
          );


        const data =
          await response
            .json()
            .catch(
              () =>
                null
            );


        if (
          !response.ok
        ) {

          throw new Error(
            data?.detail ||
            "Unable to load market news."
          );
        }


        setNewsData(
          data
        );


        setNewsPage(
          1
        );

      } catch (
        error
      ) {

        setNewsError(
          error.message ||
          "Unable to load market news."
        );

      } finally {

        setNewsLoading(
          false
        );
      }
    }


    useEffect(
      () => {

        loadNewsSentiment();

      },
      [
        selectedSymbol,
      ]
    );


    const breakdown =
      newsData?.breakdown ||
      {};


    const totalArticles =
      Number(
        newsData
          ?.total_articles_analyzed ||
        0
      );


    const positivePct =
      Number(
        breakdown
          ?.positive_percent ||
        0
      );


    const neutralPct =
      Number(
        breakdown
          ?.neutral_percent ||
        0
      );


    const negativePct =
      Number(
        breakdown
          ?.negative_percent ||
        0
      );


    const overall =
      String(
        newsData
          ?.overall_sentiment ||
        "NO_DATA"
      ).toUpperCase();


    const latestNews =
      newsData?.latest_news ||
      [];


    const maxPage =
      Math.max(
        1,
        Math.ceil(
          latestNews.length /
          PAGE_SIZE
        )
      );


    const visibleNews =
      latestNews.slice(
        (
          newsPage -
          1
        ) *
          PAGE_SIZE,
        newsPage *
          PAGE_SIZE
      );


    function sentimentClasses(
      sentiment
    ) {

      const value =
        String(
          sentiment ||
          ""
        ).toUpperCase();


      if (
        value ===
        "POSITIVE"
      ) {

        return {
          text:
            "text-green-400",
          bg:
            "bg-green-500/10",
          border:
            "border-green-500/20",
        };
      }


      if (
        value ===
        "NEGATIVE"
      ) {

        return {
          text:
            "text-red-400",
          bg:
            "bg-red-500/10",
          border:
            "border-red-500/20",
        };
      }


      return {
        text:
          "text-amber-400",
        bg:
          "bg-amber-500/10",
        border:
          "border-amber-500/20",
      };
    }


    function overallLabel() {

      if (
        overall ===
        "POSITIVE"
      ) {

        return "Positive";
      }


      if (
        overall ===
        "NEGATIVE"
      ) {

        return "Negative";
      }


      if (
        overall ===
        "NEUTRAL"
      ) {

        return "Neutral";
      }


      return "No Data";
    }


    function articleTime(
      value
    ) {

      if (
        !value
      ) {

        return "";
      }


      const date =
        new Date(
          value
        );


      if (
        Number.isNaN(
          date.getTime()
        )
      ) {

        return "";
      }


      const diffMs =
        Date.now() -
        date.getTime();


      const minutes =
        Math.max(
          1,
          Math.round(
            diffMs /
            60000
          )
        );


      if (
        minutes <
        60
      ) {

        return `${minutes}m ago`;
      }


      const hours =
        Math.round(
          minutes /
          60
        );


      if (
        hours <
        24
      ) {

        return `${hours}h ago`;
      }


      const days =
        Math.round(
          hours /
          24
        );


      return `${days}d ago`;
    }


    function sourceInitials(
      source
    ) {

      return String(
        source ||
        "N"
      )
        .split(
          /\s+/
        )
        .filter(
          Boolean
        )
        .slice(
          0,
          2
        )
        .map(
          (
            word
          ) =>
            word[
              0
            ]
        )
        .join(
          ""
        )
        .toUpperCase();
    }


    const donutStyle = {
      background:
        `conic-gradient(
          #22c55e 0deg ${positivePct * 3.6}deg,
          #f59e0b ${positivePct * 3.6}deg ${(positivePct + neutralPct) * 3.6}deg,
          #ef4444 ${(positivePct + neutralPct) * 3.6}deg 360deg
        )`,
    };


    const trendData =
      newsData?.trend ||
      [];


    const sourceRows =
      newsData?.sources ||
      [];


    const topicRows =
      newsData
        ?.trending_topics ||
      [];


    const maxSourcePercent =
      Math.max(
        1,
        ...sourceRows.map(
          (
            item
          ) =>
            Number(
              item.percent ||
              0
            )
        )
      );


    return (
      <div>

        <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-end">

          <div>

            <h1 className="text-2xl font-bold text-white">
              News & Sentiment
            </h1>

            <p className="mt-1 text-sm text-gray-500">
              Track stock news, sentiment trends, and market mood.
            </p>

            <p className="mt-1 text-[10px] text-gray-700">
              Only directly relevant recent articles are included in sentiment.
            </p>

          </div>


          <div className="flex flex-wrap items-center gap-2">

            <div className="min-w-[210px] rounded-xl border border-white/5 bg-[#0f141d] px-3 py-2">

              <p className="text-[9px] uppercase tracking-wider text-gray-600">
                Selected Stock
              </p>

              <p className="mt-1 text-xs font-semibold text-white">
                {selectedStockInfo?.short ||
                  selectedSymbol.replace(
                    ".NS",
                    ""
                  )}
              </p>

            </div>


            <button
              onClick={
                loadNewsSentiment
              }
              disabled={
                newsLoading
              }
              className="flex items-center gap-2 rounded-xl border border-violet-500/20 bg-violet-500/10 px-3 py-3 text-[11px] font-semibold text-violet-300 transition hover:bg-violet-500/15 disabled:cursor-wait disabled:opacity-60"
            >

              {newsLoading ? (
                <LoaderCircle
                  size={14}
                  className="animate-spin"
                />
              ) : (
                <RefreshCw
                  size={14}
                />
              )}

              Refresh News

            </button>

          </div>

        </div>


        {newsError && (

          <div className="mt-4 rounded-xl border border-red-500/15 bg-red-500/[0.04] px-4 py-3 text-[11px] text-red-400">
            {newsError}
          </div>

        )}


        <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-5">

          <div className="rounded-xl border border-white/5 bg-[#0f141d] p-4">

            <div className="flex items-center justify-between">

              <p className="text-[10px] text-gray-500">
                Overall Sentiment
              </p>

              <Info
                size={12}
                className="text-gray-700"
              />

            </div>

            <p className={`mt-2 text-xl font-bold ${
              sentimentClasses(
                overall
              ).text
            }`}>
              {overallLabel()}
            </p>

            <p className="mt-1 text-[10px] text-gray-600">
              Based on{" "}
              {totalArticles} relevant recent{" "}
              {totalArticles === 1
                ? "article"
                : "articles"}
            </p>

          </div>


          <div className="rounded-xl border border-green-500/10 bg-[#0f141d] p-4">

            <p className="text-[10px] text-gray-500">
              Positive News
            </p>

            <p className="mt-2 text-xl font-bold text-green-400">
              {formatPercent(
                positivePct,
                0
              )}
            </p>

            <p className="mt-1 text-[10px] text-gray-600">
              {breakdown
                ?.positive_count ||
                0} articles
            </p>

          </div>


          <div className="rounded-xl border border-amber-500/10 bg-[#0f141d] p-4">

            <p className="text-[10px] text-gray-500">
              Neutral News
            </p>

            <p className="mt-2 text-xl font-bold text-amber-400">
              {formatPercent(
                neutralPct,
                0
              )}
            </p>

            <p className="mt-1 text-[10px] text-gray-600">
              {breakdown
                ?.neutral_count ||
                0} articles
            </p>

          </div>


          <div className="rounded-xl border border-red-500/10 bg-[#0f141d] p-4">

            <p className="text-[10px] text-gray-500">
              Negative News
            </p>

            <p className="mt-2 text-xl font-bold text-red-400">
              {formatPercent(
                negativePct,
                0
              )}
            </p>

            <p className="mt-1 text-[10px] text-gray-600">
              {breakdown
                ?.negative_count ||
                0} articles
            </p>

          </div>


          <div className="rounded-xl border border-blue-500/10 bg-[#0f141d] p-4">

            <p className="text-[10px] text-gray-500">
              Articles Analyzed
            </p>

            <p className="mt-2 text-xl font-bold text-blue-400">
              {totalArticles}
            </p>

            <p className="mt-1 text-[10px] text-gray-600">
              {totalArticles} relevant of{" "}
              {newsData?.raw_articles_received ??
                totalArticles} fetched
            </p>

          </div>

        </div>


        <div className="mt-4 grid min-w-0 gap-4 xl:grid-cols-[1.55fr_0.95fr]">

          <div className="min-w-0 rounded-2xl border border-white/5 bg-[#0f141d]">

            <div className="flex items-center justify-between border-b border-white/5 px-5 py-4">

              <div className="flex items-center gap-2">

                <Newspaper
                  size={16}
                  className="text-gray-400"
                />

                <h2 className="text-sm font-semibold text-white">
                  Latest News for{" "}
                  {selectedStockInfo?.short ||
                    selectedSymbol.replace(
                      ".NS",
                      ""
                    )}
                </h2>

              </div>

              <span className="text-[10px] text-blue-400">
                {latestNews.length} loaded
              </span>

            </div>


            {newsLoading &&
            !newsData ? (

              <div className="flex min-h-[360px] items-center justify-center">

                <div className="text-center">

                  <LoaderCircle
                    size={24}
                    className="mx-auto animate-spin text-violet-400"
                  />

                  <p className="mt-3 text-xs text-gray-500">
                    Loading recent news...
                  </p>

                </div>

              </div>

            ) : visibleNews.length ===
              0 ? (

              <div className="flex min-h-[360px] items-center justify-center px-5">

                <div className="text-center">

                  <Newspaper
                    size={26}
                    className="mx-auto text-gray-700"
                  />

                  <p className="mt-3 text-sm font-medium text-gray-400">
                    No recent news available
                  </p>

                  <p className="mt-1 text-[11px] text-gray-600">
                    Try another stock or refresh later.
                  </p>

                </div>

              </div>

            ) : (

              <div>

                {visibleNews.map(
                  (
                    item,
                    index
                  ) => {

                    const classes =
                      sentimentClasses(
                        item.sentiment
                      );


                    return (
                      <a
                        key={`${item.title}-${index}`}
                        href={
                          item.url ||
                          undefined
                        }
                        target={
                          item.url
                            ? "_blank"
                            : undefined
                        }
                        rel="noreferrer"
                        className="grid gap-3 border-b border-white/5 px-5 py-4 transition hover:bg-white/[0.02] md:grid-cols-[42px_1fr_auto] md:items-start"
                      >

                        <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/5 bg-[#111925] text-[10px] font-bold text-gray-400">
                          {sourceInitials(
                            item.source
                          )}
                        </div>


                        <div className="min-w-0">

                          <p className="text-[12px] font-medium leading-5 text-gray-200">
                            {item.title}
                          </p>

                          <div className="mt-1 flex flex-wrap gap-x-2 text-[10px] text-gray-600">

                            <span>
                              {item.source}
                            </span>

                            {articleTime(
                              item
                                .published_at_utc
                            ) && (
                              <>
                                <span>
                                  •
                                </span>
                                <span>
                                  {articleTime(
                                    item
                                      .published_at_utc
                                  )}
                                </span>
                              </>
                            )}

                          </div>

                          {item.summary && (

                            <p className="mt-2 line-clamp-2 text-[10px] leading-4 text-gray-600">
                              {item.summary}
                            </p>

                          )}

                        </div>


                        <span className={`w-fit rounded-full border px-2 py-1 text-[9px] font-medium ${classes.border} ${classes.bg} ${classes.text}`}>
                          {String(
                            item.sentiment
                          ).charAt(
                            0
                          ) +
                            String(
                              item.sentiment
                            )
                              .slice(
                                1
                              )
                              .toLowerCase()}
                        </span>

                      </a>
                    );
                  }
                )}


                <div className="flex items-center justify-between px-5 py-3">

                  <p className="text-[10px] text-gray-600">
                    Showing{" "}
                    {latestNews.length
                      ? (
                          newsPage -
                          1
                        ) *
                          PAGE_SIZE +
                        1
                      : 0}
                    {" "}to{" "}
                    {Math.min(
                      newsPage *
                        PAGE_SIZE,
                      latestNews.length
                    )}
                    {" "}of{" "}
                    {latestNews.length}
                  </p>


                  <div className="flex items-center gap-2">

                    <button
                      disabled={
                        newsPage <=
                        1
                      }
                      onClick={() =>
                        setNewsPage(
                          (
                            current
                          ) =>
                            Math.max(
                              1,
                              current -
                                1
                            )
                        )
                      }
                      className="rounded-lg border border-white/5 bg-[#0b1018] p-2 text-gray-500 transition hover:text-white disabled:opacity-30"
                    >
                      <ChevronLeft
                        size={13}
                      />
                    </button>

                    <span className="text-[10px] text-gray-500">
                      {newsPage} /{" "}
                      {maxPage}
                    </span>

                    <button
                      disabled={
                        newsPage >=
                        maxPage
                      }
                      onClick={() =>
                        setNewsPage(
                          (
                            current
                          ) =>
                            Math.min(
                              maxPage,
                              current +
                                1
                            )
                        )
                      }
                      className="rounded-lg border border-white/5 bg-[#0b1018] p-2 text-gray-500 transition hover:text-white disabled:opacity-30"
                    >
                      <ChevronRight
                        size={13}
                      />
                    </button>

                  </div>

                </div>

              </div>

            )}

          </div>


          <div className="min-w-0 space-y-4">

            <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-5">

              <div className="flex items-center gap-2">

                <h2 className="text-sm font-semibold text-white">
                  Sentiment Breakdown
                </h2>

                <Info
                  size={12}
                  className="text-gray-700"
                />

              </div>


              <div className="mt-5 flex flex-col items-center gap-5 sm:flex-row">

                <div
                  className="relative h-36 w-36 shrink-0 rounded-full"
                  style={
                    donutStyle
                  }
                >

                  <div className="absolute inset-[18px] flex flex-col items-center justify-center rounded-full bg-[#0f141d]">

                    <p className="text-xl font-bold text-white">
                      {totalArticles}
                    </p>

                    <p className="text-[9px] text-gray-600">
                      Articles
                    </p>

                  </div>

                </div>


                <div className="w-full space-y-3">

                  <div className="flex items-center justify-between text-[11px]">

                    <span className="flex items-center gap-2 text-gray-400">
                      <span className="h-2 w-2 rounded-full bg-green-500" />
                      Positive
                    </span>

                    <strong className="font-semibold text-white">
                      {formatPercent(
                        positivePct,
                        0
                      )}
                    </strong>

                  </div>


                  <div className="flex items-center justify-between text-[11px]">

                    <span className="flex items-center gap-2 text-gray-400">
                      <span className="h-2 w-2 rounded-full bg-amber-500" />
                      Neutral
                    </span>

                    <strong className="font-semibold text-white">
                      {formatPercent(
                        neutralPct,
                        0
                      )}
                    </strong>

                  </div>


                  <div className="flex items-center justify-between text-[11px]">

                    <span className="flex items-center gap-2 text-gray-400">
                      <span className="h-2 w-2 rounded-full bg-red-500" />
                      Negative
                    </span>

                    <strong className="font-semibold text-white">
                      {formatPercent(
                        negativePct,
                        0
                      )}
                    </strong>

                  </div>

                </div>

              </div>

            </div>


            <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-5">

              <div className="flex items-center gap-2">

                <Flame
                  size={16}
                  className="text-orange-400"
                />

                <h2 className="text-sm font-semibold text-white">
                  Trending Topics
                </h2>

              </div>


              <div className="mt-4 flex flex-wrap gap-2">

                {topicRows.length ? (

                  topicRows.map(
                    (
                      item
                    ) => (

                      <span
                        key={
                          item.topic
                        }
                        className="rounded-full border border-violet-500/15 bg-violet-500/[0.07] px-3 py-1.5 text-[10px] font-medium text-violet-300"
                      >
                        {item.topic}
                        {" · "}
                        {item.count}
                      </span>

                    )
                  )

                ) : (

                  <p className="text-[10px] text-gray-600">
                    No dominant topics in the current feed.
                  </p>

                )}

              </div>

            </div>

          </div>

        </div>


        <div className="mt-4 grid min-w-0 gap-4 xl:grid-cols-[1.3fr_0.8fr_0.8fr]">

          <div className="min-w-0 rounded-2xl border border-white/5 bg-[#0f141d] p-5">

            <div className="flex items-center justify-between">

              <div>

                <h2 className="text-sm font-semibold text-white">
                  Sentiment Trend Over Time
                </h2>

                <p className="mt-1 text-[10px] text-gray-600">
                  Based on dates present in the current news feed.
                </p>

              </div>

              <span className="rounded-lg border border-white/5 bg-[#0b1018] px-2.5 py-1.5 text-[10px] text-gray-500">
                Recent 7 Days
              </span>

            </div>


            <div className="mt-5">

              {trendData.length ? (

                <ResponsiveContainer
                  width="100%"
                  height={260}
                >

                  <LineChart
                    data={
                      trendData
                    }
                    margin={{
                      top: 8,
                      right: 10,
                      left: -20,
                      bottom: 0,
                    }}
                  >

                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#1f2937"
                      vertical={false}
                    />

                    <XAxis
                      dataKey="date"
                      tick={{
                        fill:
                          "#64748b",
                        fontSize:
                          10,
                      }}
                      axisLine={false}
                      tickLine={false}
                    />

                    <YAxis
                      domain={[
                        0,
                        100,
                      ]}
                      tick={{
                        fill:
                          "#64748b",
                        fontSize:
                          10,
                      }}
                      axisLine={false}
                      tickLine={false}
                    />

                    <Tooltip
                      contentStyle={{
                        background:
                          "#0b1018",
                        border:
                          "1px solid #1f2937",
                        borderRadius:
                          "10px",
                        fontSize:
                          "11px",
                      }}
                    />

                    <Line
                      type="monotone"
                      dataKey="positive_percent"
                      name="Positive %"
                      stroke="#22c55e"
                      strokeWidth={2}
                      dot={{
                        r: 2,
                      }}
                    />

                    <Line
                      type="monotone"
                      dataKey="neutral_percent"
                      name="Neutral %"
                      stroke="#f59e0b"
                      strokeWidth={2}
                      dot={{
                        r: 2,
                      }}
                    />

                    <Line
                      type="monotone"
                      dataKey="negative_percent"
                      name="Negative %"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={{
                        r: 2,
                      }}
                    />

                  </LineChart>

                </ResponsiveContainer>

              ) : (

                <div className="flex h-[260px] items-center justify-center rounded-xl border border-dashed border-white/5 bg-[#0b1018]/40">

                  <p className="text-[11px] text-gray-600">
                    Not enough dated news items for a trend chart.
                  </p>

                </div>

              )}

            </div>

          </div>


          <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-5">

            <h2 className="text-sm font-semibold text-white">
              Top News Sources
            </h2>

            <div className="mt-5 space-y-4">

              {sourceRows.length ? (

                sourceRows.slice(
                  0,
                  6
                ).map(
                  (
                    item
                  ) => (

                    <div
                      key={
                        item.source
                      }
                    >

                      <div className="flex items-center justify-between text-[10px]">

                        <span className="max-w-[70%] truncate text-gray-400">
                          {item.source}
                        </span>

                        <span className="text-gray-500">
                          {formatPercent(
                            item.percent,
                            0
                          )}
                        </span>

                      </div>

                      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/5">

                        <div
                          className="h-full rounded-full bg-blue-500"
                          style={{
                            width:
                              `${Math.max(
                                4,
                                (
                                  Number(
                                    item.percent ||
                                    0
                                  ) /
                                  maxSourcePercent
                                ) *
                                  100
                              )}%`,
                          }}
                        />

                      </div>

                    </div>

                  )
                )

              ) : (

                <p className="text-[10px] text-gray-600">
                  No source distribution available.
                </p>

              )}

            </div>

          </div>


          <div className="rounded-2xl border border-violet-500/10 bg-[#0f141d] p-5">

            <div className="flex items-center gap-2">

              <Bot
                size={17}
                className="text-violet-400"
              />

              <h2 className="text-sm font-semibold text-white">
                Market Mood
              </h2>

            </div>


            <p className={`mt-5 text-lg font-bold ${
              sentimentClasses(
                overall
              ).text
            }`}>
              {overallLabel()}
            </p>


            <p className="mt-3 text-[11px] leading-5 text-gray-500">
              {newsData
                ?.market_mood_summary ||
                "No market-mood summary is available yet."}
            </p>


            <div className="mt-5 rounded-xl border border-blue-500/10 bg-blue-500/[0.03] p-3">

              <p className="text-[9px] font-semibold uppercase tracking-wider text-blue-400">
                Sentiment Method
              </p>

              <p className="mt-2 text-[10px] leading-4 text-gray-600">
                Rule-based analysis of recent Yahoo Finance headlines and summaries. It is descriptive and not a calibrated confidence score.
              </p>

            </div>


            <p className="mt-4 text-[9px] text-gray-700">
              Source:{" "}
              {newsData?.source ||
                "Yahoo Finance via yfinance"}
            </p>

          </div>

        </div>

      </div>
    );
  }


  // =======================================================
  // REFERENCE-LAYOUT PLACEHOLDER PAGES
  // =======================================================

  function PlaceholderPage({
    title,
    description,
  }) {
    return (
      <div>

        <h1 className="text-2xl font-bold text-white">
          {title}
        </h1>

        <p className="mt-1 text-sm text-gray-500">
          {description}
        </p>

        <div className="mt-6 rounded-xl border border-[#1b2738] bg-[#0f141d] p-8 text-center">
          <p className="text-sm font-medium text-gray-300">
            UI section prepared for the next feature phase.
          </p>
          <p className="mt-2 text-xs text-gray-600">
            Existing StockVision models and backend remain unchanged.
          </p>
        </div>

      </div>
    );
  }


  // =======================================================
  // PROFILE PAGE
  // =======================================================

  function ProfilePage() {

    return (
      <div>

        <div className="flex flex-col gap-5 lg:flex-row lg:items-center">

          <div className="flex h-20 w-20 items-center justify-center rounded-3xl bg-gradient-to-br from-indigo-500 to-violet-500 text-2xl font-bold text-white shadow-xl shadow-indigo-500/20">
            A
          </div>


          <div>

            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-indigo-400">
              StockVision Profile
            </p>

            <h1 className="mt-2 text-3xl font-bold text-white">
              Aryan
            </h1>

            <p className="mt-2 text-sm text-gray-500">
              B.Tech Artificial Intelligence & Machine Learning
            </p>

          </div>

        </div>


        <div className="mt-7 grid gap-4 md:grid-cols-2 xl:grid-cols-4">

          <StatCard
            title="Project"
            value="StockVision AI"
            subtitle="AI Market Intelligence"
          />

          <StatCard
            title="Frontend"
            value="React + Vite"
            subtitle="Tailwind · Charts"
          />

          <StatCard
            title="Backend"
            value="FastAPI"
            subtitle="Python · yfinance"
          />

          <StatCard
            title="AI Engines"
            value="BiLSTM + V9"
            subtitle="Forecast + Relative Strength"
          />

        </div>


        <div className="mt-6 rounded-2xl border border-white/5 bg-[#0f141d] p-6">

          <h2 className="font-semibold text-white">
            About this workspace
          </h2>

          <p className="mt-3 max-w-3xl text-sm leading-6 text-gray-500">
            StockVision AI combines live market data, real OHLC candlestick charts,
            technical indicators, X2 probabilistic next-day forecasting, V9 relative-strength
            intelligence and walk-forward model analytics in one full-stack
            market-intelligence dashboard.
          </p>


          <button
            onClick={() =>
              window.open(
                "https://github.com/Aryan2624/StockVision-AI",
                "_blank",
                "noopener,noreferrer"
              )
            }
            className="mt-5 rounded-xl bg-indigo-500 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-indigo-400"
          >
            Open GitHub Repository
          </button>

        </div>

      </div>
    );
  }


  // =======================================================
  // SETTINGS PAGE
  // =======================================================

  function SettingsPage() {

    const [
      backendCheckStatus,
      setBackendCheckStatus,
    ] = useState(
      "idle"
    );


    const [
      backendCheckMessage,
      setBackendCheckMessage,
    ] = useState(
      ""
    );


    async function checkBackendConnection() {

      try {

        setBackendCheckStatus(
          "checking"
        );

        setBackendCheckMessage(
          "Checking StockVision FastAPI..."
        );


        const [
          rootResponse,
          stocksResponse,
        ] =
          await Promise.all([
            fetch(
              `${API_URL}/`
            ),
            fetch(
              `${API_URL}/stocks`
            ),
          ]);


        if (
          !rootResponse.ok ||
          !stocksResponse.ok
        ) {

          throw new Error(
            "One or more backend services did not respond correctly."
          );
        }


        const stocksData =
          await stocksResponse.json();


        setBackendCheckStatus(
          "online"
        );

        setBackendCheckMessage(
          `Backend online · ${Number(
            stocksData?.count ||
            0
          ).toLocaleString(
            "en-IN"
          )} NSE securities available`
        );


      } catch (
        error
      ) {

        setBackendCheckStatus(
          "offline"
        );

        setBackendCheckMessage(
          error.message ||
          "Unable to connect to StockVision backend."
        );
      }
    }


    function resetPreferences() {

      setTheme(
        "dark"
      );

      setNotificationsEnabled(
        true
      );

      setNotificationsRead(
        false
      );

      setAutoRefreshEnabled(
        true
      );

      setRefreshIntervalMs(
        60000
      );

      setSelectedRange(
        "3mo"
      );
    }


    return (
      <div>

        {/* HEADER */}

        <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-end">

          <div>

            <h1 className="text-2xl font-bold text-white">
              Settings
            </h1>


            <p className="mt-1 text-sm text-gray-500">
              Control your StockVision appearance, notifications and live-data behaviour.
            </p>

          </div>


          <button
            onClick={
              resetPreferences
            }
            className="w-fit rounded-xl border border-white/10 bg-[#0f141d] px-4 py-2.5 text-xs font-medium text-gray-300 transition hover:border-violet-500/30 hover:text-violet-400"
          >
            Reset Preferences
          </button>

        </div>


        <div className="mt-6 grid gap-4 xl:grid-cols-2">

          {/* APPEARANCE */}

          <div className="rounded-2xl border border-[#1b2738] bg-[#0f141d] p-5">

            <div className="flex items-center gap-3">

              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-500/10">

                {isLightTheme ? (

                  <Sun
                    size={17}
                    className="text-violet-400"
                  />

                ) : (

                  <Moon
                    size={17}
                    className="text-violet-400"
                  />

                )}

              </div>


              <div>

                <h2 className="text-sm font-semibold text-white">
                  Appearance
                </h2>

                <p className="mt-0.5 text-[10px] text-gray-600">
                  Choose how StockVision looks.
                </p>

              </div>

            </div>


            <div className="mt-5 grid grid-cols-2 gap-3">

              <button
                onClick={() =>
                  setTheme(
                    "dark"
                  )
                }
                className={`rounded-xl border p-4 text-left transition ${
                  theme ===
                  "dark"
                    ? "border-violet-500/40 bg-violet-500/10"
                    : "border-white/5 bg-[#0b1018] hover:border-white/10"
                }`}
              >

                <Moon
                  size={18}
                  className={
                    theme ===
                    "dark"
                      ? "text-violet-400"
                      : "text-gray-500"
                  }
                />

                <p className="mt-3 text-xs font-semibold text-white">
                  Dark Mode
                </p>

                <p className="mt-1 text-[9px] text-gray-600">
                  Finance-terminal style
                </p>

              </button>


              <button
                onClick={() =>
                  setTheme(
                    "light"
                  )
                }
                className={`rounded-xl border p-4 text-left transition ${
                  theme ===
                  "light"
                    ? "border-violet-500/40 bg-violet-500/10"
                    : "border-white/5 bg-[#0b1018] hover:border-white/10"
                }`}
              >

                <Sun
                  size={18}
                  className={
                    theme ===
                    "light"
                      ? "text-violet-400"
                      : "text-gray-500"
                  }
                />

                <p className="mt-3 text-xs font-semibold text-white">
                  Light Mode
                </p>

                <p className="mt-1 text-[9px] text-gray-600">
                  Bright dashboard style
                </p>

              </button>

            </div>

          </div>


          {/* NOTIFICATIONS */}

          <div className="rounded-2xl border border-[#1b2738] bg-[#0f141d] p-5">

            <div className="flex items-start justify-between gap-4">

              <div className="flex items-center gap-3">

                <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-500/10">

                  <Bell
                    size={17}
                    className="text-blue-400"
                  />

                </div>


                <div>

                  <h2 className="text-sm font-semibold text-white">
                    Notifications
                  </h2>

                  <p className="mt-0.5 text-[10px] text-gray-600">
                    Control the header notification bell.
                  </p>

                </div>

              </div>


              <button
                onClick={() =>
                  setNotificationsEnabled(
                    (
                      current
                    ) =>
                      !current
                  )
                }
                className={`relative h-6 w-11 shrink-0 rounded-full transition ${
                  notificationsEnabled
                    ? "bg-blue-500"
                    : "bg-gray-700"
                }`}
                aria-label="Toggle notifications"
              >

                <span
                  className={`absolute top-1 h-4 w-4 rounded-full bg-white shadow transition-all ${
                    notificationsEnabled
                      ? "left-6"
                      : "left-1"
                  }`}
                />

              </button>

            </div>


            <div className="mt-5 rounded-xl border border-white/5 bg-[#0b1018] p-4">

              <div className="flex items-center justify-between">

                <span className="text-xs text-gray-400">
                  Status
                </span>

                <span
                  className={`rounded-full px-2.5 py-1 text-[9px] font-semibold ${
                    notificationsEnabled
                      ? "bg-green-500/10 text-green-400"
                      : "bg-gray-500/10 text-gray-500"
                  }`}
                >
                  {notificationsEnabled
                    ? "Enabled"
                    : "Disabled"}
                </span>

              </div>


              <p className="mt-3 text-[10px] leading-5 text-gray-600">
                When enabled, the bell can show market-data updates, AI prediction status, V9 results, forecast status and service notices.
              </p>

            </div>

          </div>


          {/* MARKET DATA */}

          <div className="rounded-2xl border border-[#1b2738] bg-[#0f141d] p-5">

            <div className="flex items-start justify-between gap-4">

              <div className="flex items-center gap-3">

                <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-green-500/10">

                  <Activity
                    size={17}
                    className="text-green-400"
                  />

                </div>


                <div>

                  <h2 className="text-sm font-semibold text-white">
                    Live Market Auto Refresh
                  </h2>

                  <p className="mt-0.5 text-[10px] text-gray-600">
                    Automatically refresh dashboard and watchlist prices.
                  </p>

                </div>

              </div>


              <button
                onClick={() =>
                  setAutoRefreshEnabled(
                    (
                      current
                    ) =>
                      !current
                  )
                }
                className={`relative h-6 w-11 shrink-0 rounded-full transition ${
                  autoRefreshEnabled
                    ? "bg-green-500"
                    : "bg-gray-700"
                }`}
                aria-label="Toggle market auto refresh"
              >

                <span
                  className={`absolute top-1 h-4 w-4 rounded-full bg-white shadow transition-all ${
                    autoRefreshEnabled
                      ? "left-6"
                      : "left-1"
                  }`}
                />

              </button>

            </div>


            <div className="mt-5">

              <p className="text-[10px] font-medium uppercase tracking-wider text-gray-600">
                Refresh Interval
              </p>


              <div className="mt-2 grid grid-cols-3 gap-2">

                {[
                  {
                    label:
                      "1 min",
                    value:
                      60000,
                  },
                  {
                    label:
                      "2 min",
                    value:
                      120000,
                  },
                  {
                    label:
                      "5 min",
                    value:
                      300000,
                  },
                ].map(
                  (
                    option
                  ) => (

                    <button
                      key={
                        option.value
                      }
                      onClick={() =>
                        setRefreshIntervalMs(
                          option.value
                        )
                      }
                      disabled={
                        !autoRefreshEnabled
                      }
                      className={`rounded-lg border px-3 py-2.5 text-[10px] font-medium transition ${
                        refreshIntervalMs ===
                          option.value &&
                        autoRefreshEnabled
                          ? "border-green-500/30 bg-green-500/10 text-green-400"
                          : "border-white/5 bg-[#0b1018] text-gray-500"
                      } ${
                        !autoRefreshEnabled
                          ? "cursor-not-allowed opacity-40"
                          : "hover:border-green-500/20"
                      }`}
                    >
                      {option.label}
                    </button>

                  )
                )}

              </div>

            </div>

          </div>


          {/* CHART PREFERENCE */}

          <div className="rounded-2xl border border-[#1b2738] bg-[#0f141d] p-5">

            <div className="flex items-center gap-3">

              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-cyan-500/10">

                <BarChart3
                  size={17}
                  className="text-cyan-400"
                />

              </div>


              <div>

                <h2 className="text-sm font-semibold text-white">
                  Chart Range
                </h2>

                <p className="mt-0.5 text-[10px] text-gray-600">
                  Select and remember the dashboard chart range.
                </p>

              </div>

            </div>


            <div className="mt-5 grid grid-cols-4 gap-2 sm:grid-cols-7">

              {RANGE_OPTIONS.map(
                (
                  option
                ) => (

                  <button
                    key={
                      option.key
                    }
                    onClick={() =>
                      setSelectedRange(
                        option.key
                      )
                    }
                    className={`rounded-lg border px-2 py-2.5 text-[10px] font-semibold transition ${
                      selectedRange ===
                      option.key
                        ? "border-cyan-500/30 bg-cyan-500/10 text-cyan-400"
                        : "border-white/5 bg-[#0b1018] text-gray-500 hover:text-gray-300"
                    }`}
                  >
                    {option.label}
                  </button>

                )
              )}

            </div>


            <p className="mt-3 text-[9px] text-gray-600">
              Current saved range:{" "}
              {RANGE_OPTIONS.find(
                (
                  item
                ) =>
                  item.key ===
                  selectedRange
              )?.label ||
                "3M"}
            </p>

          </div>

        </div>


        {/* SYSTEM / CONNECTION */}

        <div className="mt-4 rounded-2xl border border-[#1b2738] bg-[#0f141d] p-5">

          <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-center">

            <div>

              <h2 className="text-sm font-semibold text-white">
                System & Data Connection
              </h2>

              <p className="mt-1 text-[10px] text-gray-600">
                StockVision FastAPI · Yahoo Finance market data · current NSE securities universe.
              </p>

            </div>


            <button
              onClick={
                checkBackendConnection
              }
              disabled={
                backendCheckStatus ===
                "checking"
              }
              className="w-fit rounded-xl bg-blue-500/10 px-4 py-2.5 text-xs font-medium text-blue-400 transition hover:bg-blue-500/15 disabled:cursor-wait disabled:opacity-60"
            >
              {backendCheckStatus ===
              "checking"
                ? "Checking..."
                : "Test Connection"}
            </button>

          </div>


          <div className="mt-5 grid gap-3 md:grid-cols-3">

            <div className="rounded-xl border border-white/5 bg-[#0b1018] p-4">

              <p className="text-[9px] uppercase tracking-wider text-gray-600">
                API
              </p>

              <p className="mt-2 text-xs font-semibold text-white">
                {API_URL}
              </p>

            </div>


            <div className="rounded-xl border border-white/5 bg-[#0b1018] p-4">

              <p className="text-[9px] uppercase tracking-wider text-gray-600">
                NSE Universe
              </p>

              <p className="mt-2 text-xs font-semibold text-blue-400">
                {stockUniverseLoading
                  ? "Loading..."
                  : `${allStocks.length.toLocaleString(
                      "en-IN"
                    )} securities`}
              </p>

            </div>


            <div className="rounded-xl border border-white/5 bg-[#0b1018] p-4">

              <p className="text-[9px] uppercase tracking-wider text-gray-600">
                Source
              </p>

              <p className="mt-2 truncate text-xs font-semibold text-gray-300">
                {stockUniverseSource}
              </p>

            </div>

          </div>


          {backendCheckStatus !==
            "idle" && (

            <div
              className={`mt-4 rounded-xl border px-4 py-3 text-xs ${
                backendCheckStatus ===
                "online"
                  ? "border-green-500/20 bg-green-500/10 text-green-400"
                  : backendCheckStatus ===
                    "offline"
                  ? "border-red-500/20 bg-red-500/10 text-red-400"
                  : "border-blue-500/20 bg-blue-500/10 text-blue-400"
              }`}
            >
              {backendCheckMessage}
            </div>

          )}

        </div>


        {/* PROJECT INFO */}

        <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">

          {[
            {
              title:
                "Market Data",
              value:
                "Yahoo Finance",
              subtitle:
                "via FastAPI",
            },
            {
              title:
                "Price Forecast",
              value:
                "BiLSTM",
              subtitle:
                "Next-day + multi-horizon",
            },
            {
              title:
                "Relative Strength",
              value:
                "V9",
              subtitle:
                "5D vs NIFTY 50",
            },
            {
              title:
                "Frontend",
              value:
                "React + Vite",
              subtitle:
                "StockVision UI",
            },
          ].map(
            (
              item
            ) => (

              <div
                key={
                  item.title
                }
                className="rounded-xl border border-[#1b2738] bg-[#0f141d] p-4"
              >

                <p className="text-[9px] uppercase tracking-wider text-gray-600">
                  {item.title}
                </p>

                <p className="mt-2 text-sm font-semibold text-white">
                  {item.value}
                </p>

                <p className="mt-1 text-[9px] text-gray-600">
                  {item.subtitle}
                </p>

              </div>

            )
          )}

        </div>

      </div>
    );
  }


  // =======================================================
  // MAIN LAYOUT
  // =======================================================

  return (
    <div
      className={`stockvision-root min-h-screen overflow-x-hidden text-white ${
        isLightTheme
          ? "stockvision-light bg-[#f3f6fb]"
          : "bg-[#060a11]"
      }`}
    >

      <style>{`
        .stockvision-light {
          color: #0f172a !important;
        }

        .stockvision-light [class*="bg-[#060a11]"],
        .stockvision-light [class*="bg-[#070a10]"],
        .stockvision-light [class*="bg-[#080d14]"],
        .stockvision-light [class*="bg-[#0a0e15]"] {
          background: #f3f6fb !important;
        }

        .stockvision-light [class*="bg-[#0f141d]"],
        .stockvision-light [class*="bg-[#0d131e]"],
        .stockvision-light [class*="bg-[#10151f]"],
        .stockvision-light [class*="bg-[#0b1018]"],
        .stockvision-light [class*="bg-[#090e16]"],
        .stockvision-light [class*="bg-[#090d14]"],
        .stockvision-light [class*="bg-[#080d15]"] {
          background: #ffffff !important;
        }

        .stockvision-light .text-white {
          color: #0f172a !important;
        }

        .stockvision-light .text-gray-200,
        .stockvision-light .text-gray-300 {
          color: #334155 !important;
        }

        .stockvision-light .text-gray-400 {
          color: #475569 !important;
        }

        .stockvision-light .text-gray-500,
        .stockvision-light .text-gray-600,
        .stockvision-light .text-gray-700 {
          color: #64748b !important;
        }

        .stockvision-light [class*="border-white"] {
          border-color: #dbe3ef !important;
        }

        .stockvision-light [class*="border-[#1b2738]"] {
          border-color: #dbe3ef !important;
        }

        .stockvision-light input {
          color: #0f172a !important;
        }

        .stockvision-light input::placeholder {
          color: #94a3b8 !important;
        }

        .stockvision-light [class*="hover:bg-white"]:hover {
          background: rgba(15, 23, 42, 0.05) !important;
        }


        .stockvision-validation-scroll {
          min-width: 0;
          max-width: 100%;
          scrollbar-gutter: stable;
        }

        .stockvision-validation-scroll table {
          max-width: none;
        }

        @media (max-width: 1280px) {
          .stock-comparison-reference {
            zoom: 0.84 !important;
            width: 119.05% !important;
          }
        }

        @media (max-width: 1024px) {
          .stock-comparison-reference {
            zoom: 1 !important;
            width: 100% !important;
          }
        }
      `}</style>

      <div className="flex min-h-screen min-w-0">


        {/* =================================================
            SIDEBAR
        ================================================= */}

        <aside className="hidden w-[228px] shrink-0 border-r border-white/5 bg-[#080d14] p-4 lg:block">

          <div className="mb-7 flex items-center justify-between">

            <div className="flex items-center gap-3">

              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-violet-600 to-fuchsia-500 shadow-lg shadow-violet-500/15">
                <LineChartIcon
                  size={20}
                  className="text-white"
                />
              </div>

              <div>
                <p className="text-base font-bold tracking-wide text-white">
                  StockVision AI
                </p>
              </div>

            </div>

            <Menu
              size={18}
              className="text-gray-500"
            />

          </div>


          <nav className="space-y-1">

            <NavItem
              icon={LayoutDashboard}
              label="Dashboard"
              active={activePage === "Dashboard"}
              onClick={() => setActivePage("Dashboard")}
            />

            <NavItem
              icon={BarChart3}
              label="Markets"
              active={activePage === "Markets"}
              onClick={() => setActivePage("Markets")}
            />

            <NavItem
              icon={Star}
              label="Watchlist"
              active={activePage === "Watchlist"}
              onClick={() => setActivePage("Watchlist")}
            />

            <div className="relative">
              <NavItem
                icon={GitCompareArrows}
                label="Stock Comparison"
                active={activePage === "Stock Comparison"}
                onClick={() => setActivePage("Stock Comparison")}
              />
              <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 rounded-full bg-violet-500 px-1.5 py-0.5 text-[8px] font-semibold text-white">
                New
              </span>
            </div>

            <NavItem
              icon={Bot}
              label="AI Prediction"
              active={activePage === "AI Prediction"}
              onClick={() => setActivePage("AI Prediction")}
            />

            <NavItem
              icon={TrendingUp}
              label="Future Forecast"
              active={activePage === "Future Forecast"}
              onClick={() => setActivePage("Future Forecast")}
            />

            <NavItem
              icon={BarChart3}
              label="Model Analytics"
              active={activePage === "Model Analytics"}
              onClick={() => setActivePage("Model Analytics")}
            />

            <NavItem
              icon={Newspaper}
              label="News & Sentiment"
              active={activePage === "News & Sentiment"}
              onClick={() => setActivePage("News & Sentiment")}
            />

            <NavItem
              icon={Bell}
              label="Alerts"
              active={activePage === "Alerts"}
              onClick={() => setActivePage("Alerts")}
            />

            <NavItem
              icon={Settings}
              label="Settings"
              active={activePage === "Settings"}
              onClick={() => setActivePage("Settings")}
            />

          </nav>


          <div className="mt-7 rounded-xl border border-[#1b2738] bg-[#0f141d] p-3.5">

            <div className="flex items-center justify-between">

              <p className="text-xs font-semibold text-white">
                My Watchlist
              </p>

              <button
                onClick={() =>
                  setActivePage(
                    "Watchlist"
                  )
                }
                className="text-[9px] font-medium text-blue-400"
              >
                Edit
              </button>

            </div>


            <div className="mt-3 space-y-1">

              {watchlistItems.slice(
                0,
                5
              ).map(
                (
                  item
                ) => {

                  const live =
                    watchlistLiveData[
                      item.symbol
                    ];


                  const move =
                    Number(
                      live?.change_percent
                    );


                  return (
                    <button
                      key={
                        `sidebar-${item.symbol}`
                      }
                      onClick={() => {
                        selectStock(
                          item.symbol
                        );
                        setActivePage(
                          "Dashboard"
                        );
                      }}
                      className={`grid w-full grid-cols-[1fr_auto] items-center gap-2 rounded-lg px-2 py-2 text-left transition ${
                        item.symbol ===
                        selectedSymbol
                          ? "bg-indigo-500/10"
                          : "hover:bg-white/[0.035]"
                      }`}
                    >

                      <div className="min-w-0">
                        <p className="truncate text-[10px] font-semibold text-gray-300">
                          {item.short}
                        </p>
                        <p
                          className={`mt-0.5 text-[9px] ${
                            Number.isFinite(
                              move
                            )
                              ? move >= 0
                                ? "text-green-400"
                                : "text-red-400"
                              : "text-gray-700"
                          }`}
                        >
                          {Number.isFinite(
                            move
                          )
                            ? formatPercent(
                                move
                              )
                            : "--"}
                        </p>
                      </div>

                      <p className="text-[10px] font-medium text-gray-400">
                        {live
                          ? formatPrice(
                              live.price
                            )
                          : "--"}
                      </p>

                    </button>
                  );
                }
              )}

            </div>

            <button
              onClick={() =>
                setActivePage(
                  "Watchlist"
                )
              }
              className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg border border-white/5 bg-white/[0.025] py-2.5 text-[10px] font-medium text-gray-400 transition hover:border-indigo-500/20 hover:text-indigo-300"
            >
              <span className="text-base leading-none">
                +
              </span>
              Add / Manage Stocks
            </button>


          </div>




        </aside>


        {/* =================================================
            MAIN AREA
        ================================================= */}

        <main className="min-w-0 flex-1">


          {/* TOP HEADER */}

          <header className="sticky top-0 z-20 h-16 border-b border-white/5 bg-[#060a11]/95 px-5 backdrop-blur-xl">

            <div className="flex h-full items-center justify-between gap-4">

              <div
                ref={searchContainerRef}
                className="relative w-full max-w-[420px]"
              >

                <div className="flex items-center rounded-lg border border-white/[0.07] bg-[#0f141d] px-3">

                  <Search
                    size={15}
                    className="text-gray-500"
                  />

                  <input
                    value={search}
                    onChange={(event) => {
                      setSearch(event.target.value);
                      setSearchOpen(true);
                    }}
                    onFocus={() => setSearchOpen(true)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        handleSearchEnter();
                      }

                      if (event.key === "Escape") {
                        setSearchOpen(false);
                      }
                    }}
                    placeholder="Search stock (e.g. RELIANCE, TCS)"
                    className="w-full bg-transparent px-3 py-2.5 text-xs text-white outline-none placeholder:text-gray-600"
                  />

                  <span className="rounded-md border border-white/5 bg-white/[0.03] px-2 py-1 text-[9px] text-gray-600">
                    Ctrl K
                  </span>

                </div>


                {searchOpen && (

                  <div className="absolute left-0 right-0 top-12 z-50 overflow-hidden rounded-xl border border-white/10 bg-[#10151f] shadow-2xl">

                    {filteredStocks.map(
                      (
                        item
                      ) => (

                        <div
                          key={
                            item.symbol
                          }
                          className="flex items-center gap-2 px-2 py-1"
                        >

                          <button
                            onClick={() =>
                              selectStock(
                                item.symbol
                              )
                            }
                            className="flex min-w-0 flex-1 items-center justify-between rounded-lg px-2 py-2 text-left transition hover:bg-white/5"
                          >

                            <div className="min-w-0">

                              <p className="truncate text-sm font-medium text-white">
                                {item.name}
                              </p>

                              <p className="text-xs text-gray-500">
                                {item.symbol}
                              </p>

                            </div>


                            <span className="ml-3 shrink-0 text-xs text-blue-400">
                              Select
                            </span>

                          </button>


                          <button
                            onClick={() =>
                              toggleWatchlist(
                                item
                              )
                            }
                            title={
                              isWatchlisted(
                                item.symbol
                              )
                                ? "Remove from watchlist"
                                : "Add to watchlist"
                            }
                            className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg transition ${
                              isWatchlisted(
                                item.symbol
                              )
                                ? "bg-yellow-500/10 text-yellow-400"
                                : "text-gray-600 hover:bg-yellow-500/10 hover:text-yellow-400"
                            }`}
                          >

                            <Star
                              size={15}
                              fill={
                                isWatchlisted(
                                  item.symbol
                                )
                                  ? "currentColor"
                                  : "none"
                              }
                            />

                          </button>

                        </div>

                      )
                    )}

                  </div>

                )}

              </div>


              <div className="flex items-center gap-4">

                {/* THEME TOGGLE */}

                <button
                  onClick={() => {

                    setTheme(
                      (
                        current
                      ) =>
                        current ===
                        "dark"
                          ? "light"
                          : "dark"
                    );

                    setNotificationOpen(
                      false
                    );

                    setProfileOpen(
                      false
                    );

                  }}
                  title={
                    isLightTheme
                      ? "Switch to dark mode"
                      : "Switch to light mode"
                  }
                  className="flex h-9 w-9 items-center justify-center rounded-lg text-gray-400 transition hover:bg-white/5 hover:text-white"
                >

                  {isLightTheme ? (

                    <Sun
                      size={17}
                    />

                  ) : (

                    <Moon
                      size={17}
                    />

                  )}

                </button>


                {/* NOTIFICATION BELL */}

                <div className="relative">

                  <button
                    onClick={() => {

                      setNotificationOpen(
                        (
                          current
                        ) =>
                          !current
                      );

                      setProfileOpen(
                        false
                      );

                    }}
                    title={
                      notificationsEnabled
                        ? "Notifications"
                        : "Notifications are disabled in Settings"
                    }
                    disabled={
                      !notificationsEnabled
                    }
                    className={`relative flex h-9 w-9 items-center justify-center rounded-lg transition ${
                      notificationsEnabled
                        ? "text-gray-400 hover:bg-white/5 hover:text-white"
                        : "cursor-not-allowed text-gray-700 opacity-60"
                    }`}
                  >

                    <Bell
                      size={17}
                    />


                    {unreadNotificationCount >
                      0 && (

                      <span className="absolute right-0.5 top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-violet-500 px-1 text-[8px] font-bold text-white shadow-lg shadow-violet-500/20">

                        {unreadNotificationCount >
                        9
                          ? "9+"
                          : unreadNotificationCount}

                      </span>

                    )}

                  </button>


                  {notificationOpen && (

                    <div className="absolute right-0 top-11 z-[100] w-[340px] overflow-hidden rounded-2xl border border-white/10 bg-[#0d131e] shadow-[0_24px_70px_rgba(0,0,0,0.45)]">

                      <div className="flex items-center justify-between border-b border-white/5 px-4 py-3.5">

                        <div>

                          <p className="text-sm font-semibold text-white">
                            Notifications
                          </p>

                          <p className="mt-0.5 text-[10px] text-gray-600">

                            {headerNotifications.length}
                            {" "}
                            StockVision updates

                          </p>

                        </div>


                        {headerNotifications.length >
                          0 && (

                          <button
                            onClick={() =>
                              setNotificationsRead(
                                true
                              )
                            }
                            className="text-[10px] font-medium text-blue-400 hover:text-blue-300"
                          >
                            Mark all as read
                          </button>

                        )}

                      </div>


                      <div className="max-h-[360px] overflow-y-auto p-2">

                        {headerNotifications.length ===
                          0 ? (

                          <div className="px-4 py-8 text-center">

                            <Bell
                              size={24}
                              className="mx-auto text-gray-700"
                            />

                            <p className="mt-3 text-xs font-medium text-gray-400">
                              No notifications
                            </p>

                            <p className="mt-1 text-[10px] text-gray-600">
                              New StockVision updates will appear here.
                            </p>

                          </div>

                        ) : (

                          headerNotifications.map(
                            (
                              item
                            ) => (

                              <div
                                key={
                                  item.id
                                }
                                className="flex gap-3 rounded-xl px-3 py-3 transition hover:bg-white/[0.04]"
                              >

                                <div
                                  className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
                                    item.type ===
                                    "warning"
                                      ? "bg-red-500/10"
                                      : item.type ===
                                        "v9"
                                      ? "bg-blue-500/10"
                                      : item.type ===
                                        "ai"
                                      ? "bg-purple-500/10"
                                      : "bg-green-500/10"
                                  }`}
                                >

                                  {item.type ===
                                  "warning" ? (

                                    <XCircle
                                      size={15}
                                      className="text-red-400"
                                    />

                                  ) : item.type ===
                                    "v9" ? (

                                    <Gauge
                                      size={15}
                                      className="text-blue-400"
                                    />

                                  ) : item.type ===
                                    "ai" ? (

                                    <Bot
                                      size={15}
                                      className="text-purple-400"
                                    />

                                  ) : item.type ===
                                    "forecast" ? (

                                    <TrendingUp
                                      size={15}
                                      className="text-green-400"
                                    />

                                  ) : (

                                    <Activity
                                      size={15}
                                      className="text-green-400"
                                    />

                                  )}

                                </div>


                                <div className="min-w-0 flex-1">

                                  <p className="text-[11px] font-semibold text-gray-200">
                                    {item.title}
                                  </p>

                                  <p className="mt-1 text-[10px] leading-4 text-gray-500">
                                    {item.message}
                                  </p>

                                </div>


                                {!notificationsRead && (

                                  <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400" />

                                )}

                              </div>

                            )
                          )

                        )}

                      </div>

                    </div>

                  )}

                </div>

                <div className="h-6 w-px bg-white/5" />

                <div className="relative">

                  <button
                    onClick={() => {

                      setProfileOpen(
                        (
                          current
                        ) =>
                          !current
                      );

                      setNotificationOpen(
                        false
                      );

                    }}
                    className="flex items-center gap-3 rounded-xl px-2 py-1.5 text-left transition hover:bg-white/[0.04]"
                    title="Open profile"
                  >

                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-indigo-500/25 text-xs font-semibold text-indigo-200">
                      A
                    </div>

                    <div className="hidden sm:block">

                      <p className="text-[11px] font-medium text-white">
                        Aryan
                      </p>

                      <p className="text-[9px] text-gray-600">
                        AIML Project
                      </p>

                    </div>

                    <ChevronRight
                      size={13}
                      className={`hidden text-gray-600 transition sm:block ${
                        profileOpen
                          ? "rotate-90"
                          : ""
                      }`}
                    />

                  </button>


                  {profileOpen && (

                    <div className="absolute right-0 top-12 z-[110] w-[290px] overflow-hidden rounded-2xl border border-white/10 bg-[#0d131e] shadow-[0_24px_70px_rgba(0,0,0,0.45)]">

                      <div className="border-b border-white/5 p-4">

                        <div className="flex items-center gap-3">

                          <div className="flex h-11 w-11 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-violet-500 text-sm font-bold text-white shadow-lg shadow-indigo-500/20">
                            A
                          </div>

                          <div>

                            <p className="text-sm font-semibold text-white">
                              Aryan
                            </p>

                            <p className="mt-0.5 text-[10px] text-gray-500">
                              B.Tech Artificial Intelligence & Machine Learning
                            </p>

                          </div>

                        </div>


                        <div className="mt-4 rounded-xl border border-blue-500/10 bg-blue-500/[0.04] px-3 py-2.5">

                          <p className="text-[10px] text-gray-500">
                            Current Project
                          </p>

                          <p className="mt-1 text-xs font-medium text-blue-400">
                            StockVision AI
                          </p>

                        </div>

                      </div>


                      <div className="p-2">

                        <button
                          onClick={() => {

                            setActivePage(
                              "Profile"
                            );

                            setProfileOpen(
                              false
                            );

                          }}
                          className="flex w-full items-center justify-between rounded-xl px-3 py-3 text-left transition hover:bg-white/[0.04]"
                        >

                          <div>

                            <p className="text-[11px] font-medium text-gray-200">
                              Profile Overview
                            </p>

                            <p className="mt-0.5 text-[9px] text-gray-600">
                              View project profile and stack
                            </p>

                          </div>

                          <ChevronRight
                            size={14}
                            className="text-gray-600"
                          />

                        </button>


                        <button
                          onClick={() => {

                            setActivePage(
                              "Settings"
                            );

                            setProfileOpen(
                              false
                            );

                          }}
                          className="flex w-full items-center justify-between rounded-xl px-3 py-3 text-left transition hover:bg-white/[0.04]"
                        >

                          <div>

                            <p className="text-[11px] font-medium text-gray-200">
                              Settings
                            </p>

                            <p className="mt-0.5 text-[9px] text-gray-600">
                              StockVision system information
                            </p>

                          </div>

                          <Settings
                            size={14}
                            className="text-gray-600"
                          />

                        </button>


                        <button
                          onClick={() => {

                            window.open(
                              "https://github.com/Aryan2624/StockVision-AI",
                              "_blank",
                              "noopener,noreferrer"
                            );

                            setProfileOpen(
                              false
                            );

                          }}
                          className="flex w-full items-center justify-between rounded-xl px-3 py-3 text-left transition hover:bg-white/[0.04]"
                        >

                          <div>

                            <p className="text-[11px] font-medium text-gray-200">
                              GitHub Repository
                            </p>

                            <p className="mt-0.5 text-[9px] text-gray-600">
                              Open StockVision AI source code
                            </p>

                          </div>

                          <ChevronRight
                            size={14}
                            className="text-gray-600"
                          />

                        </button>

                      </div>

                    </div>

                  )}

                </div>

              </div>

            </div>

          </header>


          {/* =================================================
              PAGE
          ================================================= */}

          <div className="p-3 lg:p-3.5 xl:p-3.5">

            {activePage ===
              "Dashboard" && (
              <DashboardPage />
            )}


            {activePage ===
              "Watchlist" && (
              <WatchlistPage />
            )}


            {activePage ===
              "Markets" && (
              <MarketsPage />
            )}


            {activePage ===
              "AI Prediction" && (
              <AIPredictionPage />
            )}


            {activePage ===
              "Future Forecast" && (
              <FutureForecastPage />
            )}


            {activePage ===
              "Model Analytics" && (
              <ModelAnalyticsPage />
            )}


            {activePage ===
              "Stock Comparison" && (
              <StockComparisonPage />
            )}


            {activePage ===
              "News & Sentiment" && (
              <NewsSentimentPage />
            )}


            {activePage ===
              "Alerts" && (
              <AlertsPage />
            )}


            {activePage ===
              "Profile" && (
              <ProfilePage />
            )}


            {activePage ===
              "Settings" && (
              <SettingsPage />
            )}

          </div>

        </main>

      </div>

    </div>
  );
}