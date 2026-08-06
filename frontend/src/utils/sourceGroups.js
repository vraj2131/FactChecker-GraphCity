// Shared source-group definitions for the landing page and top-bar input.

export const SOURCE_GROUPS = [
  { key: 'wikipedia',  label: 'Wikipedia',          desc: 'FAISS index + live Wikipedia',      alwaysOn: true  },
  { key: 'live_news',  label: 'Live News',           desc: 'Guardian · NewsAPI · GDELT'                         },
  { key: 'factcheck',  label: 'Fact-Checkers',       desc: 'Professional fact-check sources'                    },
  { key: 'scientific', label: 'Scientific',          desc: 'OpenAlex · PubMed · arXiv'                         },
  { key: 'financial',  label: 'Financial / Crypto',  desc: 'FRED · SEC · CoinGecko · World Bank'               },
  { key: 'web_search', label: 'Web Search',          desc: 'DuckDuckGo — may slow results'                      },
  { key: 'social',     label: 'Social Media',        desc: 'Reddit + Bluesky'                                   },
];

export const DEFAULT_GROUPS = ['wikipedia', 'live_news', 'factcheck', 'scientific', 'financial'];

// True when the selection matches the backend defaults exactly. Callers pass
// null to the API in that case so App.jsx's graph-cache fast-path stays usable.
export function isDefaultSelection(enabledGroups) {
  return (
    enabledGroups.size === DEFAULT_GROUPS.length &&
    DEFAULT_GROUPS.every((g) => enabledGroups.has(g))
  );
}
