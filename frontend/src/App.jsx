import {
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  Activity,
  Bell,
  Bot,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  Circle,
  Gauge,
  LayoutDashboard,
  LineChart,
  LoaderCircle,
  Search,
  Settings,
  Star,
  TrendingDown,
  TrendingUp,
  Wallet,
  XCircle,
} from "lucide-react";

import {
  ResponsiveContainer,
  ComposedChart,
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


const WATCHLIST =
  STOCKS.slice(
    0,
    6
  );


const RANGE_OPTIONS = [
  {
    key: "1d",
    label: "1D",
  },
  {
    key: "5d",
    label: "5D",
  },
  {
    key: "1mo",
    label: "1M",
  },
  {
    key: "6mo",
    label: "6M",
  },
  {
    key: "ytd",
    label: "YTD",
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
    <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-5">

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
    <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-5">

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
    <div className="h-[390px] w-full">

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
    <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-5 transition hover:border-white/10">

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
        text="Running BiLSTM prediction..."
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


  const move =
    Number(
      prediction.predicted_return_percent ||
      0
    );


  const signal =
    prediction.trend_signal ||
    (
      move >= 0.25
        ? "BULLISH"
        : move <= -0.25
        ? "BEARISH"
        : "NEUTRAL"
    );


  return (
    <div>

      <div className="flex flex-wrap items-center justify-between gap-3">

        <div>

          <p className="text-xs text-gray-500">
            Expected Next-Day Move
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
          title="Predicted Next Close"
          value={
            formatPrice(
              prediction.predicted_price
            )
          }
        />


        <StatCard
          title="Price Direction"
          value={
            prediction.direction
          }
        />


        <StatCard
          title="Model"
          value={
            prediction.model
          }
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
    selectedRange,
    setSelectedRange,
  ] = useState(
    "1d"
  );


  const [
    stock,
    setStock,
  ] = useState(
    null
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
  // SELECTED STOCK INFO
  // =======================================================

  const selectedStockInfo =
    STOCKS.find(
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

          return STOCKS.slice(
            0,
            7
          );
        }


        return STOCKS.filter(
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
      ]
    );


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
  // MARKET DATA EFFECT
  // =======================================================

  useEffect(
    () => {

      fetchStock(
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

      const interval =
        setInterval(
          () => {

            fetchStock(
              selectedSymbol,
              selectedRange
            );

          },
          60000
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
      STOCKS.find(
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
      STOCKS.find(
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

    return (
      <>

        <div className="mb-6 flex flex-col justify-between gap-5 xl:flex-row xl:items-end">

          <div>

            <div className="flex flex-wrap items-center gap-3">

              <h1 className="text-2xl font-bold text-white">
                {stock?.symbol ||
                  selectedSymbol}
              </h1>


              <span className="rounded-full bg-blue-500/10 px-2.5 py-1 text-xs text-blue-400">
                NSE
              </span>

            </div>


            <p className="mt-1 text-sm text-gray-500">
              {selectedStockInfo.name}
            </p>

          </div>


          <div className="xl:text-right">

            <div className="flex items-center gap-2 xl:justify-end">

              <span className="h-2 w-2 rounded-full bg-green-400" />

              <span className="text-xs font-medium text-green-400">
                CONNECTED
              </span>

            </div>


            <p className="mt-2 text-3xl font-bold text-white">

              {stockLoading &&
              !stock
                ? "Loading..."
                : formatPrice(
                    currentPrice
                  )}

            </p>


            {stock && (

              <div
                className={`mt-1 flex items-center gap-1 text-sm font-medium xl:justify-end ${
                  positiveDay
                    ? "text-green-400"
                    : "text-red-400"
                }`}
              >

                {positiveDay
                  ? (
                    <TrendingUp
                      size={15}
                    />
                  )
                  : (
                    <TrendingDown
                      size={15}
                    />
                  )}


                <span>

                  {change >= 0
                    ? "+"
                    : ""}

                  {change.toFixed(
                    2
                  )}

                </span>


                <span>

                  (
                  {formatPercent(
                    changePercent
                  )}
                  )

                </span>

              </div>

            )}


            <p className="mt-1 text-[11px] text-gray-600">
              NSE · Delayed Quote · INR
            </p>

          </div>

        </div>


        {stockError && (

          <div className="mb-5 rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-400">
            {stockError}
          </div>

        )}


        {/* MARKET CHART */}

        <div className="mb-6 rounded-2xl border border-white/5 bg-[#0f141d] p-5">

          <div className="mb-4 flex flex-col justify-between gap-4 sm:flex-row sm:items-center">

            <div>

              <p className="font-semibold text-white">
                Market Price
              </p>

              <p className="mt-1 text-xs text-gray-500">
                Historical and latest market data
              </p>

            </div>


            <div className="flex flex-wrap gap-1 rounded-xl bg-[#0a0f17] p-1">

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
                    className={`rounded-lg px-3 py-2 text-xs font-medium transition ${
                      selectedRange ===
                      range.key

                        ? "bg-blue-500 text-white"

                        : "text-gray-500 hover:bg-white/5 hover:text-gray-300"
                    }`}
                  >
                    {range.label}
                  </button>

                )
              )}

            </div>

          </div>


          {stockLoading &&
          !stock ? (

            <LoadingBox
              text="Loading market chart..."
            />

          ) : (

            <YahooStyleChart
              stock={
                stock
              }
              selectedRange={
                selectedRange
              }
            />

          )}

        </div>


        {/* MARKET CARDS */}

        <div className="mb-7 grid grid-cols-2 gap-4 lg:grid-cols-4">

          <StatCard
            title="Open"
            value={
              formatPrice(
                stock?.open
              )
            }
          />

          <StatCard
            title="High"
            value={
              formatPrice(
                stock?.high
              )
            }
          />

          <StatCard
            title="Low"
            value={
              formatPrice(
                stock?.low
              )
            }
          />

          <StatCard
            title="Volume"
            value={
              formatVolume(
                stock?.volume
              )
            }
          />

        </div>


        {/* INDICATORS */}

        <div className="mb-7">

          <div className="mb-4 flex items-center gap-2">

            <Activity
              size={18}
              className="text-blue-400"
            />

            <h2 className="text-lg font-semibold text-white">
              Technical Indicators
            </h2>

          </div>


          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">

            <IndicatorCard
              title="RSI (14)"
              value={
                rsiValue !==
                  null &&
                rsiValue !==
                  undefined

                  ? Number(
                      rsiValue
                    ).toFixed(
                      2
                    )

                  : "--"
              }
              status={
                rsiStatus
              }
            />


            <IndicatorCard
              title="MACD"
              value={
                macdValue !==
                  null &&
                macdValue !==
                  undefined

                  ? Number(
                      macdValue
                    ).toFixed(
                      2
                    )

                  : "--"
              }
              status={
                macdStatus
              }
            />


            <IndicatorCard
              title="SMA 20"
              value={
                formatPrice(
                  sma20Value
                )
              }
            />


            <IndicatorCard
              title="EMA 20"
              value={
                formatPrice(
                  ema20Value
                )
              }
            />

          </div>

        </div>


        {/* AI CARDS */}

        <div className="grid gap-6 xl:grid-cols-3">

          <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-6">

            <div className="mb-6 flex items-center justify-between">

              <div className="flex items-center gap-3">

                <div className="rounded-xl bg-purple-500/10 p-2.5">

                  <Bot
                    size={20}
                    className="text-purple-400"
                  />

                </div>


                <div>

                  <h2 className="font-semibold text-white">
                    AI Next-Day Prediction
                  </h2>

                  <p className="text-xs text-gray-500">
                    BiLSTM return forecast
                  </p>

                </div>

              </div>


              <button
                onClick={() =>
                  setActivePage(
                    "AI Prediction"
                  )
                }
                className="text-xs text-blue-400"
              >
                Details
              </button>

            </div>


            <PredictionContent
              prediction={
                prediction
              }
              loading={
                predictionLoading
              }
              error={
                predictionError
              }
            />

          </div>


          <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-6">

            <div className="mb-6 flex items-center justify-between">

              <div className="flex items-center gap-3">

                <div className="rounded-xl bg-blue-500/10 p-2.5">

                  <Gauge
                    size={20}
                    className="text-blue-400"
                  />

                </div>


                <div>

                  <h2 className="font-semibold text-white">
                    AI Relative Strength
                  </h2>

                  <p className="text-xs text-gray-500">
                    V9 · 5D vs NIFTY 50
                  </p>

                </div>

              </div>


              <button
                onClick={() =>
                  setActivePage(
                    "AI Prediction"
                  )
                }
                className="text-xs text-blue-400"
              >
                Details
              </button>

            </div>


            <RelativeStrengthContent
              data={
                relativePrediction
              }
              loading={
                relativeLoading
              }
              error={
                relativeError
              }
              compact
            />

          </div>


          <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-6">

            <div className="mb-5 flex items-center justify-between">

              <div>

                <h2 className="font-semibold text-white">
                  Future Forecast
                </h2>

                <p className="mt-1 text-xs text-gray-500">
                  1D · 3D · 5D · 10D
                </p>

              </div>


              <TrendingUp
                size={20}
                className="text-blue-400"
              />

            </div>


            <div className="rounded-xl border border-blue-500/10 bg-blue-500/5 p-5">

              <p className="text-sm font-medium text-white">
                Multi-Horizon BiLSTM
              </p>


              <p className="mt-2 text-sm leading-6 text-gray-500">

                Forecast future percentage movement with
                validation-based ranges across multiple trading horizons.

              </p>


              <button
                onClick={() =>
                  setActivePage(
                    "Future Forecast"
                  )
                }
                className="mt-5 flex items-center gap-2 rounded-xl bg-blue-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-400"
              >

                Open Future Forecast

                <ChevronRight
                  size={16}
                />

              </button>

            </div>

          </div>

        </div>

      </>
    );
  }


  // =======================================================
  // WATCHLIST
  // =======================================================

  function WatchlistPage() {

    return (
      <div>

        <h1 className="text-2xl font-bold text-white">
          Watchlist
        </h1>


        <p className="mt-1 text-sm text-gray-500">
          Quickly open tracked NSE stocks.
        </p>


        <div className="mt-6 grid gap-4 md:grid-cols-2">

          {WATCHLIST.map(
            (
              item
            ) => (

              <button
                key={
                  item.symbol
                }
                onClick={() => {

                  selectStock(
                    item.symbol
                  );

                  setActivePage(
                    "Dashboard"
                  );

                }}
                className="rounded-2xl border border-white/5 bg-[#0f141d] p-5 text-left transition hover:border-blue-500/30"
              >

                <div className="flex items-center justify-between">

                  <div>

                    <p className="font-semibold text-white">
                      {item.short}
                    </p>

                    <p className="mt-1 text-xs text-gray-500">
                      {item.name}
                    </p>

                  </div>


                  <Star
                    size={18}
                    className="text-yellow-400"
                  />

                </div>

              </button>

            )
          )}

        </div>

      </div>
    );
  }


  // =======================================================
  // MARKETS
  // =======================================================

  function MarketsPage() {

    return (
      <div>

        <h1 className="text-2xl font-bold text-white">
          Markets
        </h1>


        <p className="mt-1 text-sm text-gray-500">
          Popular NSE stocks.
        </p>


        <div className="mt-6 overflow-hidden rounded-2xl border border-white/5 bg-[#0f141d]">

          {STOCKS.map(
            (
              item
            ) => (

              <div
                key={
                  item.symbol
                }
                className="flex items-center justify-between border-b border-white/5 px-5 py-4 last:border-0"
              >

                <div>

                  <p className="text-sm font-medium text-white">
                    {item.name}
                  </p>

                  <p className="mt-1 text-xs text-gray-500">
                    {item.symbol}
                  </p>

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
                  className="rounded-lg bg-blue-500/10 px-3 py-2 text-xs text-blue-400"
                >
                  View
                </button>

              </div>

            )
          )}

        </div>

      </div>
    );
  }


  // =======================================================
  // AI PREDICTION PAGE
  // =======================================================

  function AIPredictionPage() {

    return (
      <div>

        <div className="mb-6 flex items-center justify-between">

          <div>

            <h1 className="text-2xl font-bold text-white">
              AI Prediction
            </h1>

            <p className="mt-1 text-sm text-gray-500">
              Next-day BiLSTM forecast for{" "}
              {selectedStockInfo.short}.
            </p>

          </div>


          <button
            onClick={() =>
              fetchPrediction(
                selectedSymbol
              )
            }
            className="rounded-xl bg-purple-500/10 px-4 py-2 text-xs font-medium text-purple-400"
          >
            Refresh Prediction
          </button>

        </div>


        <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-6">

          <PredictionContent
            prediction={
              prediction
            }
            loading={
              predictionLoading
            }
            error={
              predictionError
            }
          />

        </div>


        <div className="mt-6 rounded-2xl border border-white/5 bg-[#0f141d] p-6">

          <div className="mb-6 flex flex-col justify-between gap-4 sm:flex-row sm:items-center">

            <div className="flex items-center gap-3">

              <div className="rounded-xl bg-blue-500/10 p-2.5">

                <Gauge
                  size={20}
                  className="text-blue-400"
                />

              </div>


              <div>

                <h2 className="font-semibold text-white">
                  V9 Relative Strength Intelligence
                </h2>

                <p className="mt-1 text-xs text-gray-500">
                  5-day performance classification relative to NIFTY 50
                </p>

              </div>

            </div>


            <button
              onClick={() =>
                fetchRelativePrediction(
                  selectedSymbol
                )
              }
              className="rounded-xl bg-blue-500/10 px-4 py-2 text-xs font-medium text-blue-400"
            >
              Refresh V9
            </button>

          </div>


          <RelativeStrengthContent
            data={
              relativePrediction
            }
            loading={
              relativeLoading
            }
            error={
              relativeError
            }
          />


          {relativePrediction?.walk_forward_evaluation && (

            <div className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">

              <StatCard
                title="Walk-Forward Accuracy"
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
                subtitle="3-class relative target"
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
                subtitle="Across all classes"
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
                subtitle="Walk-forward evaluation"
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
                subtitle="Historical benchmark"
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
                subtitle="Relative-momentum baseline"
              />

            </div>

          )}

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

            <div className="mb-6 rounded-2xl border border-white/5 bg-[#0f141d] p-5">

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

              <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-6">

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


              <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-6">

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

        <div className="mb-6 rounded-2xl border border-white/5 bg-[#0f141d] p-5">

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

        <div className="mb-6 rounded-2xl border border-white/5 bg-[#0f141d] p-5">

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

          <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-5">

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


          <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-5">

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
  // SETTINGS PAGE
  // =======================================================

  function SettingsPage() {

    return (
      <div>

        <h1 className="text-2xl font-bold text-white">
          Settings
        </h1>


        <p className="mt-1 text-sm text-gray-500">
          StockVision system information.
        </p>


        <div className="mt-6 space-y-4">

          <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-5">

            <p className="font-medium text-white">
              Market Data
            </p>

            <p className="mt-1 text-sm text-gray-500">
              Yahoo Finance through StockVision FastAPI.
            </p>

          </div>


          <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-5">

            <p className="font-medium text-white">
              AI Forecasting
            </p>

            <p className="mt-1 text-sm text-gray-500">

              Next-day BiLSTM + Multi-Horizon 1D / 3D / 5D / 10D BiLSTM.

            </p>

          </div>


          <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-5">

            <p className="font-medium text-white">
              V9 Relative Strength Intelligence
            </p>

            <p className="mt-1 text-sm text-gray-500">

              Predicts 5-day UNDERPERFORM / NEUTRAL / OUTPERFORM
              behavior relative to NIFTY 50 using the final V9
              production model.

            </p>

          </div>


          <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-5">

            <p className="font-medium text-white">
              Model Analytics
            </p>

            <p className="mt-1 text-sm text-gray-500">
              V9 walk-forward testing, baseline comparison, per-stock robustness, non-overlapping checks and feature importance.
            </p>

          </div>


          <div className="rounded-2xl border border-white/5 bg-[#0f141d] p-5">

            <p className="font-medium text-white">
              Automatic Training
            </p>

            <p className="mt-1 text-sm text-gray-500">

              Supported new NSE tickers automatically receive a saved
              Multi-Horizon model the first time they are forecast.

            </p>

          </div>

        </div>

      </div>
    );
  }


  // =======================================================
  // MAIN LAYOUT
  // =======================================================

  return (
    <div className="min-h-screen bg-[#080b12] text-white">

      <div className="flex min-h-screen">


        {/* =================================================
            SIDEBAR
        ================================================= */}

        <aside className="hidden w-64 shrink-0 border-r border-white/5 bg-[#0a0e15] p-5 lg:block">

          <div className="mb-10 flex items-center gap-3 px-2">

            <div className="rounded-xl bg-blue-500/10 p-2">

              <LineChart
                size={22}
                className="text-blue-400"
              />

            </div>


            <div>

              <p className="font-bold tracking-wide text-white">
                StockVision
              </p>

              <p className="text-[10px] text-gray-500">
                AI MARKET INTELLIGENCE
              </p>

            </div>

          </div>


          <nav className="space-y-2">

            <NavItem
              icon={
                LayoutDashboard
              }
              label="Dashboard"
              active={
                activePage ===
                "Dashboard"
              }
              onClick={() =>
                setActivePage(
                  "Dashboard"
                )
              }
            />


            <NavItem
              icon={
                Star
              }
              label="Watchlist"
              active={
                activePage ===
                "Watchlist"
              }
              onClick={() =>
                setActivePage(
                  "Watchlist"
                )
              }
            />


            <NavItem
              icon={
                Activity
              }
              label="Markets"
              active={
                activePage ===
                "Markets"
              }
              onClick={() =>
                setActivePage(
                  "Markets"
                )
              }
            />


            <NavItem
              icon={
                Bot
              }
              label="AI Prediction"
              active={
                activePage ===
                "AI Prediction"
              }
              onClick={() =>
                setActivePage(
                  "AI Prediction"
                )
              }
            />


            <NavItem
              icon={
                TrendingUp
              }
              label="Future Forecast"
              active={
                activePage ===
                "Future Forecast"
              }
              onClick={() =>
                setActivePage(
                  "Future Forecast"
                )
              }
            />


            <NavItem
              icon={
                BarChart3
              }
              label="Model Analytics"
              active={
                activePage ===
                "Model Analytics"
              }
              onClick={() =>
                setActivePage(
                  "Model Analytics"
                )
              }
            />


            <NavItem
              icon={
                Settings
              }
              label="Settings"
              active={
                activePage ===
                "Settings"
              }
              onClick={() =>
                setActivePage(
                  "Settings"
                )
              }
            />

          </nav>


          <div className="mt-10 rounded-2xl border border-green-500/10 bg-green-500/5 p-4">

            <div className="flex items-center gap-2">

              <span className="h-2 w-2 rounded-full bg-green-400" />

              <span className="text-xs text-green-400">
                Backend Connected
              </span>

            </div>


            <p className="mt-2 text-xs leading-5 text-gray-500">

              Live market analytics and AI services connected.

            </p>

          </div>

        </aside>


        {/* =================================================
            MAIN AREA
        ================================================= */}

        <main className="min-w-0 flex-1">


          {/* TOP HEADER */}

          <header className="sticky top-0 z-20 border-b border-white/5 bg-[#080b12]/95 px-5 py-4 backdrop-blur-xl lg:px-8">

            <div className="flex items-center justify-between gap-4">


              {/* SEARCH */}

              <div className="relative w-full max-w-xl">

                <div className="flex items-center rounded-xl border border-white/5 bg-[#0f141d] px-4">

                  <Search
                    size={17}
                    className="text-gray-500"
                  />


                  <input
                    value={
                      search
                    }
                    onChange={
                      (
                        event
                      ) => {

                        setSearch(
                          event.target.value
                        );

                        setSearchOpen(
                          true
                        );

                      }
                    }
                    onFocus={() =>
                      setSearchOpen(
                        true
                      )
                    }
                    onKeyDown={
                      (
                        event
                      ) => {

                        if (
                          event.key ===
                          "Enter"
                        ) {

                          handleSearchEnter();

                        }


                        if (
                          event.key ===
                          "Escape"
                        ) {

                          setSearchOpen(
                            false
                          );

                        }

                      }
                    }
                    placeholder="Search NSE ticker... RELIANCE, WIPRO, MARUTI"
                    className="w-full bg-transparent px-3 py-3 text-sm text-white outline-none placeholder:text-gray-600"
                  />

                </div>


                {searchOpen && (

                  <div className="absolute left-0 right-0 top-14 z-50 overflow-hidden rounded-xl border border-white/10 bg-[#10151f] shadow-2xl">

                    {filteredStocks.map(
                      (
                        item
                      ) => (

                        <button
                          key={
                            item.symbol
                          }
                          onClick={() =>
                            selectStock(
                              item.symbol
                            )
                          }
                          className="flex w-full items-center justify-between px-4 py-3 text-left transition hover:bg-white/5"
                        >

                          <div>

                            <p className="text-sm font-medium text-white">
                              {item.name}
                            </p>

                            <p className="text-xs text-gray-500">
                              {item.symbol}
                            </p>

                          </div>


                          <span className="text-xs text-blue-400">
                            Select
                          </span>

                        </button>

                      )
                    )}


                    {search.trim() &&
                    filteredStocks.length ===
                      0 && (

                      <button
                        onClick={() =>
                          selectStock(
                            search
                          )
                        }
                        className="flex w-full items-center justify-between border-t border-white/5 px-4 py-3 text-left hover:bg-white/5"
                      >

                        <div>

                          <p className="text-sm font-medium text-white">
                            Use NSE ticker
                          </p>

                          <p className="text-xs text-gray-500">
                            {normalizeStockSymbol(
                              search
                            )}
                          </p>

                        </div>


                        <Search
                          size={15}
                          className="text-blue-400"
                        />

                      </button>

                    )}

                  </div>

                )}

              </div>


              {/* USER */}

              <div className="hidden items-center gap-4 sm:flex">

                <Bell
                  size={19}
                  className="text-gray-400"
                />


                <div className="flex items-center gap-3 border-l border-white/5 pl-4">

                  <div className="flex h-9 w-9 items-center justify-center rounded-full bg-blue-500/10">

                    <Wallet
                      size={17}
                      className="text-blue-400"
                    />

                  </div>


                  <div>

                    <p className="text-xs font-medium text-white">
                      Investor
                    </p>

                    <p className="text-[10px] text-gray-500">
                      StockVision AI
                    </p>

                  </div>

                </div>

              </div>

            </div>

          </header>


          {/* =================================================
              PAGE
          ================================================= */}

          <div className="p-5 lg:p-8">

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
              "Settings" && (
              <SettingsPage />
            )}

          </div>

        </main>

      </div>

    </div>
  );
}