import React, { useMemo } from "react";
import Chart from "react-apexcharts";


function formatIndianPrice(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "--";
  }

  return `₹${number.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}


function formatCompactVolume(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "--";
  }

  if (number >= 10000000) {
    return `${(number / 10000000).toFixed(2)} Cr`;
  }

  if (number >= 100000) {
    return `${(number / 100000).toFixed(2)} L`;
  }

  if (number >= 1000) {
    return `${(number / 1000).toFixed(2)} K`;
  }

  return number.toLocaleString("en-IN");
}


export default function CandlestickStockChart({
  candles = [],
  currentPrice,
  previousClose,
  rangeLabel = "1D",
  loading = false,
  error = "",
  theme = "dark",
}) {
  const isLight =
    theme === "light";
  const prepared = useMemo(() => {
    return (candles || [])
      .map((item) => {
        const x = new Date(item.timestamp).getTime();

        const open = Number(item.open);
        const high = Number(item.high);
        const low = Number(item.low);
        const close = Number(item.close);
        const volume = Number(item.volume || 0);

        if (
          !Number.isFinite(x) ||
          !Number.isFinite(open) ||
          !Number.isFinite(high) ||
          !Number.isFinite(low) ||
          !Number.isFinite(close)
        ) {
          return null;
        }

        return {
          x,
          open,
          high,
          low,
          close,
          volume,
          positive: close >= open,
        };
      })
      .filter(Boolean);
  }, [candles]);


  const candleSeries = useMemo(
    () => [
      {
        name: "Price",
        data: prepared.map((item) => ({
          x: item.x,
          y: [
            item.open,
            item.high,
            item.low,
            item.close,
          ],
        })),
      },
    ],
    [prepared]
  );


  const volumeSeries = useMemo(
    () => [
      {
        name: "Volume",
        data: prepared.map((item) => ({
          x: item.x,
          y: item.volume,
          fillColor: item.positive
            ? "rgba(16,185,129,0.52)"
            : "rgba(239,68,68,0.52)",
        })),
      },
    ],
    [prepared]
  );


  const priceAnnotations = [];

  const numericCurrentPrice = Number(currentPrice);
  const numericPreviousClose = Number(previousClose);

  if (Number.isFinite(numericPreviousClose)) {
    priceAnnotations.push({
      y: numericPreviousClose,
      borderColor: "rgba(148,163,184,0.50)",
      strokeDashArray: 6,
      label: {
        show: false,
      },
    });
  }

  if (Number.isFinite(numericCurrentPrice)) {
    priceAnnotations.push({
      y: numericCurrentPrice,
      borderColor: numericCurrentPrice >= numericPreviousClose
        ? "#10b981"
        : "#ef4444",
      strokeDashArray: 0,
      label: {
        borderColor: numericCurrentPrice >= numericPreviousClose
          ? "#10b981"
          : "#ef4444",
        style: {
          background: numericCurrentPrice >= numericPreviousClose
            ? "#10b981"
            : "#ef4444",
          color: "#ffffff",
          fontSize: "10px",
          fontWeight: 600,
        },
        text: Number.isFinite(numericCurrentPrice)
          ? numericCurrentPrice.toFixed(2)
          : "",
        position: "right",
        offsetX: 2,
      },
    });
  }


  const priceOptions = {
    chart: {
      id: "stockvision-candle-price",
      group: "stockvision-market",
      type: "candlestick",
      background: "transparent",
      toolbar: {
        show: false,
      },
      zoom: {
        enabled: false,
      },
      animations: {
        enabled: false,
      },
      foreColor: "#64748b",
      fontFamily: "inherit",
    },

    theme: {
      mode: isLight
        ? "light"
        : "dark",
    },

    plotOptions: {
      candlestick: {
        colors: {
          upward: "#10b981",
          downward: "#ef4444",
        },
        wick: {
          useFillColor: true,
        },
      },
    },

    grid: {
      borderColor: isLight
        ? "rgba(148,163,184,0.30)"
        : "rgba(51,65,85,0.42)",
      strokeDashArray: 0,
      xaxis: {
        lines: {
          show: true,
        },
      },
      yaxis: {
        lines: {
          show: true,
        },
      },
      padding: {
        left: 2,
        right: 6,
        top: 0,
        bottom: -4,
      },
    },

    xaxis: {
      type: "datetime",
      tooltip: {
        enabled: false,
      },
      axisBorder: {
        show: false,
      },
      axisTicks: {
        show: false,
      },
      labels: {
        datetimeUTC: false,
        hideOverlappingLabels: true,
        style: {
          colors: isLight
            ? "#64748b"
            : "#64748b",
          fontSize: "10px",
        },
        formatter: (value, timestamp) => {
          const date = new Date(timestamp);

          if (rangeLabel === "1D") {
            return date.toLocaleTimeString("en-IN", {
              hour: "2-digit",
              minute: "2-digit",
              hour12: false,
            });
          }

          if (rangeLabel === "1W") {
            return date.toLocaleDateString("en-IN", {
              weekday: "short",
            });
          }

          return date.toLocaleDateString("en-IN", {
            day: "2-digit",
            month: "short",
          });
        },
      },
      crosshairs: {
        show: true,
        stroke: {
          color: "#475569",
          width: 1,
          dashArray: 4,
        },
      },
    },

    yaxis: {
      opposite: true,
      decimalsInFloat: 2,
      labels: {
        minWidth: 52,
        style: {
          colors: "#64748b",
          fontSize: "10px",
        },
        formatter: (value) =>
          Number(value).toLocaleString("en-IN", {
            maximumFractionDigits: 0,
          }),
      },
      tooltip: {
        enabled: true,
      },
    },

    tooltip: {
      enabled: true,
      shared: false,
      theme: "dark",
      custom: ({ seriesIndex, dataPointIndex, w }) => {
        const point = prepared[dataPointIndex];

        if (!point) {
          return "";
        }

        const move = point.close - point.open;
        const movePercent = point.open
          ? (move / point.open) * 100
          : 0;

        const moveClass =
          move >= 0
            ? "#4ade80"
            : "#fb7185";

        const date = new Date(point.x).toLocaleString("en-IN", {
          day: "2-digit",
          month: "short",
          year: "numeric",
          hour: rangeLabel === "1D" ? "2-digit" : undefined,
          minute: rangeLabel === "1D" ? "2-digit" : undefined,
          hour12: false,
        });

        return `
          <div style="
            min-width:210px;
            padding:12px;
            background:${isLight ? "#ffffff" : "#090e16"};
            border:1px solid ${isLight ? "rgba(15,23,42,.12)" : "rgba(255,255,255,.10)"};
            border-radius:12px;
            box-shadow:0 18px 45px rgba(0,0,0,.35);
            font-family:inherit;
          ">
            <div style="font-size:11px;color:#94a3b8;margin-bottom:9px;">
              <span style="color:${isLight ? "#475569" : "#94a3b8"};">${date}</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:7px;font-size:11px;">
              <span style="color:#64748b;">Open</span>
              <b style="color:#e2e8f0;text-align:right;">${formatIndianPrice(point.open)}</b>

              <span style="color:#64748b;">High</span>
              <b style="color:#4ade80;text-align:right;">${formatIndianPrice(point.high)}</b>

              <span style="color:#64748b;">Low</span>
              <b style="color:#fb7185;text-align:right;">${formatIndianPrice(point.low)}</b>

              <span style="color:#64748b;">Close</span>
              <b style="color:#e2e8f0;text-align:right;">${formatIndianPrice(point.close)}</b>

              <span style="color:#64748b;">Volume</span>
              <b style="color:#e2e8f0;text-align:right;">${formatCompactVolume(point.volume)}</b>

              <span style="color:#64748b;">Move</span>
              <b style="color:${moveClass};text-align:right;">
                ${move >= 0 ? "+" : ""}${movePercent.toFixed(2)}%
              </b>
            </div>
          </div>
        `;
      },
    },

    annotations: {
      yaxis: priceAnnotations,
    },

    stroke: {
      width: 1,
    },

    noData: {
      text: "No candlestick data",
      align: "center",
      verticalAlign: "middle",
      style: {
        color: "#64748b",
        fontSize: "12px",
      },
    },
  };


  const volumeOptions = {
    chart: {
      id: "stockvision-candle-volume",
      group: "stockvision-market",
      type: "bar",
      background: "transparent",
      toolbar: {
        show: false,
      },
      zoom: {
        enabled: false,
      },
      animations: {
        enabled: false,
      },
      foreColor: "#64748b",
      fontFamily: "inherit",
    },

    theme: {
      mode: isLight
        ? "light"
        : "dark",
    },

    plotOptions: {
      bar: {
        columnWidth: "72%",
        borderRadius: 0,
      },
    },

    dataLabels: {
      enabled: false,
    },

    grid: {
      borderColor: isLight
        ? "rgba(148,163,184,0.20)"
        : "rgba(51,65,85,0.28)",
      strokeDashArray: 0,
      xaxis: {
        lines: {
          show: true,
        },
      },
      yaxis: {
        lines: {
          show: false,
        },
      },
      padding: {
        top: -13,
        left: 2,
        right: 6,
        bottom: -8,
      },
    },

    xaxis: {
      type: "datetime",
      tooltip: {
        enabled: false,
      },
      axisBorder: {
        show: false,
      },
      axisTicks: {
        show: false,
      },
      labels: {
        show: false,
        datetimeUTC: false,
      },
      crosshairs: {
        show: true,
        stroke: {
          color: "#475569",
          width: 1,
          dashArray: 4,
        },
      },
    },

    yaxis: {
      opposite: true,
      labels: {
        show: false,
      },
    },

    tooltip: {
      enabled: false,
    },
  };


  if (loading) {
    return (
      <div className="flex h-[330px] items-center justify-center text-sm text-gray-500">
        Loading candlestick chart...
      </div>
    );
  }


  if (error) {
    return (
      <div className="flex h-[330px] items-center justify-center p-6 text-center">
        <div>
          <p className="text-sm font-medium text-red-400">
            Candlestick chart unavailable
          </p>
          <p className="mt-2 text-xs text-gray-600">
            {error}
          </p>
        </div>
      </div>
    );
  }


  if (prepared.length === 0) {
    return (
      <div className="flex h-[330px] items-center justify-center text-sm text-gray-500">
        No candlestick data available.
      </div>
    );
  }


  return (
    <div className="w-full overflow-hidden">

      <div className="h-[255px] w-full">
        <Chart
          options={priceOptions}
          series={candleSeries}
          type="candlestick"
          height="100%"
          width="100%"
        />
      </div>


      <div className="-mt-1 h-[74px] w-full border-t border-white/[0.025]">
        <Chart
          options={volumeOptions}
          series={volumeSeries}
          type="bar"
          height="100%"
          width="100%"
        />
      </div>

    </div>
  );
}
