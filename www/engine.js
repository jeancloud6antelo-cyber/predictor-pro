/* Motor de predicción: Poisson/Dixon-Coles + Elo + forma (fútbol) y Elo + forma + H2H (esports) */
(function (root) {
  const E = {};
  const fact = [1];
  for (let i = 1; i <= 20; i++) fact[i] = fact[i - 1] * i;
  const pois = (k, l) => Math.exp(-l) * Math.pow(l, k) / fact[k];
  const avg = (a, d = 0.5) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : d);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

  function tau(x, y, lh, la, rho) {
    if (x === 0 && y === 0) return 1 - lh * la * rho;
    if (x === 0 && y === 1) return 1 + lh * rho;
    if (x === 1 && y === 0) return 1 + la * rho;
    if (x === 1 && y === 1) return 1 - rho;
    return 1;
  }

  E.scoreMatrix = function (lh, la, rho = -0.06, max = 8) {
    const m = [];
    let tot = 0;
    for (let x = 0; x <= max; x++) {
      m[x] = [];
      for (let y = 0; y <= max; y++) {
        const p = Math.max(0, pois(x, lh) * pois(y, la) * tau(x, y, lh, la, rho));
        m[x][y] = p;
        tot += p;
      }
    }
    for (let x = 0; x <= max; x++) for (let y = 0; y <= max; y++) m[x][y] /= tot;
    return m;
  };

  E.buildFootball = function (fin, now = Date.now(), opt = {}) {
    const halfLife = opt.halfLife || 70; // días
    const k = opt.shrink || 5;
    const lam = Math.LN2 / (halfLife * 864e5);
    const T = {};
    const get = (n) => T[n] || (T[n] = { n, gh: 0, ch: 0, wh: 0, ga: 0, ca: 0, wa: 0, N: 0, elo: 1500, form: [] });
    fin = fin.slice().sort((a, b) => a.t - b.t);
    let sH = 0, sA = 0, sw = 0;
    for (const m of fin) {
      const w = Math.exp(-lam * Math.max(0, now - m.t));
      const h = get(m.home), a = get(m.away);
      h.gh += w * m.hg; h.ch += w * m.ag; h.wh += w;
      a.ga += w * m.ag; a.ca += w * m.hg; a.wa += w;
      h.N++; a.N++;
      sH += w * m.hg; sA += w * m.ag; sw += w;
      const d = m.hg - m.ag;
      const Eh = 1 / (1 + Math.pow(10, -((h.elo + 60) - a.elo) / 400));
      const S = d > 0 ? 1 : d < 0 ? 0 : 0.5;
      const g = Math.max(1, Math.log(Math.abs(d) + 1.7));
      const dl = 20 * g * (S - Eh);
      h.elo += dl; a.elo -= dl;
      h.form.push(S); a.form.push(1 - S);
    }
    const muH = sw ? sH / sw : 1.45, muA = sw ? sA / sw : 1.15;

    function predict(home, away) {
      const h = T[home], a = T[away];
      if (!h || !a) return null;
      const sh = (h.gh + k * muH) / (h.wh + k);
      const ch = (h.ch + k * muA) / (h.wh + k);
      const sa = (a.ga + k * muA) / (a.wa + k);
      const ca = (a.ca + k * muH) / (a.wa + k);
      const fh = avg(h.form.slice(-5)), fa = avg(a.form.slice(-5));
      let lh = (sh * ca) / muH * (1 + 0.1 * (fh - 0.5));
      let la = (sa * ch) / muA * (1 + 0.1 * (fa - 0.5));
      lh = clamp(lh, 0.15, 5); la = clamp(la, 0.15, 5);
      const M = E.scoreMatrix(lh, la);
      let pH = 0, pD = 0, pA = 0, o25 = 0, btts = 0, best = [0, 0, -1];
      for (let x = 0; x < M.length; x++)
        for (let y = 0; y < M[x].length; y++) {
          const p = M[x][y];
          if (x > y) pH += p; else if (x === y) pD += p; else pA += p;
          if (x + y >= 3) o25 += p;
          if (x > 0 && y > 0) btts += p;
          if (p > best[2]) best = [x, y, p];
        }
      // mezcla con Elo (30%)
      const Ee = 1 / (1 + Math.pow(10, -((h.elo + 60) - a.elo) / 400));
      const eH = (1 - pD) * Ee, eA = (1 - pD) * (1 - Ee);
      const bH = 0.7 * pH + 0.3 * eH, bA = 0.7 * pA + 0.3 * eA, bD = pD;
      const s = bH + bD + bA;
      const probs = [bH / s, bD / s, bA / s];
      const pickIdx = probs.indexOf(Math.max(...probs));
      const n = Math.min(h.N, a.N);
      const score = clamp(n / 20, 0, 1) * 0.5 + clamp((Math.max(...probs) - 0.34) / 0.4, 0, 1) * 0.5;
      return {
        probs, pickIdx, lh, la, over25: o25, btts, score: [best[0], best[1]],
        conf: score > 0.66 ? 'Alta' : score > 0.4 ? 'Media' : 'Baja', confScore: score,
        info: { eloH: h.elo, eloA: a.elo, formH: fh, formA: fa, n },
      };
    }
    return { predict, teams: T };
  };

  E.buildEsports = function (past, now = Date.now()) {
    const T = {}, H = {};
    const g = (n) => T[n] || (T[n] = { n, elo: 1500, N: 0, form: [] });
    past = past.slice().sort((x, y) => x.t - y.t);
    for (const m of past) {
      if (!m.winner) continue;
      const A = g(m.a), B = g(m.b);
      const ea = 1 / (1 + Math.pow(10, (B.elo - A.elo) / 400));
      const S = m.winner === m.a ? 1 : 0;
      const mg = 1 + 0.15 * clamp(Math.abs((m.sa || 0) - (m.sb || 0)) - 1, 0, 3);
      const d = 24 * mg * (S - ea);
      A.elo += d; B.elo -= d; A.N++; B.N++;
      A.form.push(S); B.form.push(1 - S);
      const key = [m.a, m.b].sort().join('|');
      (H[key] = H[key] || []).push(m.winner);
    }
    function predict(a, b) {
      const A = T[a], B = T[b];
      if (!A || !B) return null;
      const pe = 1 / (1 + Math.pow(10, (B.elo - A.elo) / 400));
      const fa = avg(A.form.slice(-5)), fb = avg(B.form.slice(-5));
      const arr = (H[[a, b].sort().join('|')] || []).slice(-6);
      const h2h = arr.length ? (arr.filter((w) => w === a).length - arr.filter((w) => w === b).length) / arr.length : 0;
      const n = Math.min(A.N, B.N);
      let logit = Math.log(pe / (1 - pe)) + 0.6 * (fa - fb) + 0.25 * h2h;
      logit *= n / (n + 4);
      const p = 1 / (1 + Math.exp(-logit));
      const probs = [p, 1 - p];
      const sc = clamp(n / 15, 0, 1) * 0.5 + clamp((Math.max(...probs) - 0.5) / 0.3, 0, 1) * 0.5;
      return {
        probs, pickIdx: p >= 0.5 ? 0 : 1,
        conf: sc > 0.66 ? 'Alta' : sc > 0.4 ? 'Media' : 'Baja', confScore: sc,
        info: { eloA: A.elo, eloB: B.elo, formA: fa, formB: fb, h2h, h2hN: arr.length, n },
      };
    }
    return { predict, teams: T };
  };

  // Valor esperado y stake (Kelly fraccionado 25%, tope 3% del bank)
  E.value = function (p, odds) {
    if (!odds || odds <= 1) return null;
    const edge = p * odds - 1;
    const kelly = edge > 0 ? edge / (odds - 1) : 0;
    return { edge, stake: Math.min(0.03, kelly * 0.25) };
  };

  // Backtest walk-forward: predice cada partido usando solo datos previos
  E.backtest = function (kind, data, frac = 0.7) {
    data = data.slice().sort((a, b) => a.t - b.t);
    const start = Math.max(30, Math.floor(data.length * frac));
    let n = 0, hit = 0, br = 0, ll = 0;
    for (let i = start; i < data.length; i++) {
      const m = data[i];
      let p, o;
      if (kind === 'f') {
        p = E.buildFootball(data.slice(0, i), m.t).predict(m.home, m.away);
        o = m.hg > m.ag ? 0 : m.hg === m.ag ? 1 : 2;
      } else {
        if (!m.winner) continue;
        p = E.buildEsports(data.slice(0, i), m.t).predict(m.a, m.b);
        o = m.winner === m.a ? 0 : 1;
      }
      if (!p) continue;
      n++;
      if (p.pickIdx === o) hit++;
      br += p.probs.reduce((s, q, j) => s + Math.pow(q - (j === o ? 1 : 0), 2), 0);
      ll += -Math.log(Math.max(1e-9, p.probs[o]));
    }
    return { n, hit: n ? hit / n : 0, brier: n ? br / n : 0, logloss: n ? ll / n : 0, baseBrier: kind === 'f' ? 2 / 3 : 0.5 };
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = E;
  else root.Engine = E;
})(typeof window !== 'undefined' ? window : globalThis);
