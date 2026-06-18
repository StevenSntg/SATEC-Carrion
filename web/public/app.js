/* SATEC — mapa de riesgo de brotes de Carrión por provincia del Perú. */
(function () {
  "use strict";

  var NIVEL = {
    alto:  "oklch(0.63 0.20 25)",
    medio: "oklch(0.80 0.15 75)",
    bajo:  "oklch(0.72 0.14 155)"
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

  var hint = document.getElementById("detail-hint");
  var card = document.getElementById("detail-card");
  var el = function (id) { return document.getElementById(id); };

  var ubigeo = function (f) { return String(f.properties.FIRST_IDPR).padStart(4, "0"); };

  Promise.all([
    fetch("data/provincias.geojson").then(function (r) { return r.json(); }),
    fetch("data/riesgo.json").then(function (r) { return r.json(); })
  ]).then(function (res) {
    var geo = res[0], riesgo = res[1];

    function styleFor(f) {
      var info = riesgo[ubigeo(f)];
      return { className: "prov-path", color: "oklch(0.16 0.012 195)",
               weight: 1, fillColor: nivelColor(info), fillOpacity: info ? 0.8 : 0.3 };
    }

    function showDetail(f) {
      var info = riesgo[ubigeo(f)];
      card.hidden = false; hint.hidden = true;
      el("detail-depto").textContent = f.properties.FIRST_NOMB || "Perú";
      el("detail-prov").textContent = f.properties.NOMBPROV || "Provincia";
      var badge = el("detail-nivel");
      if (info) {
        badge.textContent = info.nivel;
        badge.className = "nivel-badge nivel-" + info.nivel;
        el("detail-rn").textContent = Math.round(info.prob_rn * 100) + "%";
        el("detail-ad").textContent = info.pred_ad ? "brote" : "sin brote";
        el("detail-casos").textContent = info.casos;
      } else {
        badge.textContent = "sin dato"; badge.className = "nivel-badge";
        el("detail-rn").textContent = "—"; el("detail-ad").textContent = "—"; el("detail-casos").textContent = "—";
      }
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
          mouseover: function () { lyr.setStyle({ weight: 2.5, color: "oklch(0.97 0.008 195)", fillOpacity: 0.95 }); lyr.bringToFront(); showDetail(f); },
          mouseout: function () { layer.resetStyle(lyr); },
          click: function () { showDetail(f); map.flyToBounds(lyr.getBounds(), { maxZoom: 9, padding: [70, 70], duration: 0.6 }); }
        });
      }
    }).addTo(map);

    map.fitBounds(layer.getBounds(), { padding: [20, 20] });
  }).catch(function (err) {
    hint.textContent = "No se pudieron cargar los datos del mapa (" + err + ").";
  });
})();
