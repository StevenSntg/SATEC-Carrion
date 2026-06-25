/* SATEC — mapa de riesgo de brotes de Carrión por provincia del Perú. */
(function () {
  "use strict";

  var NIVEL = {
    alto:  "oklch(0.64 0.20 25)",
    medio: "oklch(0.80 0.15 75)",
    bajo:  "oklch(0.74 0.14 155)"
  };
  var NA = "oklch(0.55 0.02 200)";
  var nivelColor = function (info) { return info && NIVEL[info.nivel] ? NIVEL[info.nivel] : NA; };

  var map = L.map("map", {
    zoomControl: false, attributionControl: true,
    scrollWheelZoom: false, minZoom: 4, maxZoom: 11
  }).setView([-9.8, -75.4], 5.4);
  L.control.zoom({ position: "bottomright" }).addTo(map);

  // Mapa base real del Perú: da el contexto geográfico (costa, fronteras, ciudades).
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &middot; &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: "abcd", maxZoom: 19
  }).addTo(map);

  var el = function (id) { return document.getElementById(id); };
  var intro = el("detail-intro");
  var card = el("detail-card");
  var probFill = el("prob-fill");
  var probMarker = el("prob-marker");
  var fixedUbigeo = null;

  var ubigeo = function (f) { return String(f.properties.FIRST_IDPR).padStart(4, "0"); };

  function renderSummary() {
    if (intro) intro.hidden = false;
    if (card) card.hidden = true;
  }

  function showDetail(f, riesgo) {
    var info = riesgo[ubigeo(f)];
    intro.hidden = true; card.hidden = false;
    el("detail-depto").textContent = f.properties.FIRST_NOMB || "Perú";
    el("detail-prov").textContent = f.properties.NOMBPROV || "Provincia";
    var badge = el("detail-nivel");

    if (info) {
      var p = Math.max(0, Math.min(1, info.prob_rn));
      badge.textContent = info.nivel;
      badge.className = "nivel-badge nivel-" + info.nivel;
      el("detail-rn").textContent = Math.round(p * 100) + "%";
      probFill.style.width = (p * 100) + "%";
      probFill.style.background = nivelColor(info);
      probMarker.style.left = "calc(" + (p * 100) + "% - 1px)";
      var ad = el("detail-ad");
      ad.textContent = info.pred_ad ? "brote" : "sin brote";
      ad.className = "pred-chip" + (info.pred_ad ? " is-brote" : "");
      el("detail-casos").textContent = info.casos;
    } else {
      badge.textContent = "sin dato"; badge.className = "nivel-badge";
      el("detail-rn").textContent = "—";
      probFill.style.width = "0%"; probMarker.style.left = "0%";
      el("detail-ad").textContent = "—"; el("detail-ad").className = "pred-chip";
      el("detail-casos").textContent = "—";
    }
  }

  Promise.all([
    fetch("data/provincias.geojson").then(function (r) { return r.json(); }),
    fetch("data/riesgo.json").then(function (r) { return r.json(); })
  ]).then(function (res) {
    var geo = res[0], riesgo = res[1];

    // Resumen de niveles (estado de la semana más reciente).
    var counts = { alto: 0, medio: 0, bajo: 0 };
    Object.keys(riesgo).forEach(function (k) {
      var n = riesgo[k].nivel;
      if (counts[n] !== undefined) counts[n] += 1;
    });
    el("count-alto").textContent = counts.alto;
    el("count-medio").textContent = counts.medio;
    el("count-bajo").textContent = counts.bajo;
    el("stat-total").textContent = Object.keys(riesgo).length;
    el("stat-alerta").textContent = counts.alto + counts.medio;

    function styleFor(f) {
      var info = riesgo[ubigeo(f)];
      return { className: "prov-path", color: "oklch(0.16 0.012 195)",
               weight: 1, fillColor: nivelColor(info), fillOpacity: info ? 0.82 : 0.28 };
    }

    var layer = L.geoJSON(geo, {
      style: styleFor,
      onEachFeature: function (f, lyr) {
        var info = riesgo[ubigeo(f)];
        var nivel = info ? info.nivel : "sin dato";
        var prob = info ? Math.round(info.prob_rn * 100) + "%" : "n/d";
        lyr.bindTooltip(
          "<b>" + (f.properties.NOMBPROV || "") + "</b><br>" +
          "<span class='tt-nivel' style='color:" + nivelColor(info) + "'>Riesgo " + nivel + "</span> &middot; RN " + prob,
          { sticky: true, className: "prov-tooltip", direction: "top", opacity: 1 });
        lyr.on({
          mouseover: function () {
            lyr.setStyle({ weight: 2.5, color: "oklch(0.97 0.008 195)", fillOpacity: 0.96 });
            lyr.bringToFront();
            showDetail(f, riesgo);
          },
          mouseout: function () {
            layer.resetStyle(lyr);
            if (fixedUbigeo) {
              var ff = fixed(geo, fixedUbigeo);
              if (ff) showDetail(ff, riesgo);
            } else {
              renderSummary();
            }
          },
          click: function () {
            fixedUbigeo = ubigeo(f);
            showDetail(f, riesgo);
            map.flyToBounds(lyr.getBounds(), { maxZoom: 9, padding: [70, 70], duration: 0.6 });
          },
          keydown: function (e) {
            if (e.originalEvent && (e.originalEvent.key === "Enter" || e.originalEvent.key === " ")) {
              fixedUbigeo = ubigeo(f); showDetail(f, riesgo);
            }
          }
        });
      }
    }).addTo(map);

    function fixed(g, ub) {
      var found = null;
      g.features.forEach(function (ft) { if (ubigeo(ft) === ub) found = ft; });
      return found;
    }

    map.fitBounds(layer.getBounds(), { padding: [20, 20] });
  }).catch(function (err) {
    var hint = el("detail-hint");
    if (hint) hint.textContent = "No se pudieron cargar los datos del mapa (" + err + ").";
  });
})();
