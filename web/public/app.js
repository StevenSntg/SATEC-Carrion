/* SATEC — mapa de riesgo de brotes de la enfermedad de Carrión por provincia. */
(function () {
  "use strict";

  var NIVEL_COLOR = { alto: "#e0514a", medio: "#e6a53b", bajo: "#34a577" };
  var NA_COLOR = "#38465f";

  var map = L.map("map", {
    zoomControl: false,
    attributionControl: false,
    scrollWheelZoom: false,
  }).setView([-9.6, -76], 6);
  L.control.zoom({ position: "bottomright" }).addTo(map);

  map.createPane("provincias");

  var hint = document.getElementById("detail-hint");
  var card = document.getElementById("detail-card");
  var elDepto = document.getElementById("detail-depto");
  var elProv = document.getElementById("detail-prov");
  var elNivel = document.getElementById("detail-nivel");
  var elRN = document.getElementById("detail-rn");
  var elAD = document.getElementById("detail-ad");
  var elCasos = document.getElementById("detail-casos");

  function ubigeo(feature) {
    return String(feature.properties.FIRST_IDPR).padStart(4, "0");
  }

  Promise.all([
    fetch("data/provincias.geojson").then(function (r) { return r.json(); }),
    fetch("data/riesgo.json").then(function (r) { return r.json(); }),
  ]).then(function (res) {
    var geo = res[0];
    var riesgo = res[1];

    function styleFor(feature) {
      var info = riesgo[ubigeo(feature)];
      var color = info ? (NIVEL_COLOR[info.nivel] || NA_COLOR) : NA_COLOR;
      return {
        pane: "provincias",
        className: "prov-path",
        color: "#0b1220",
        weight: 1,
        fillColor: color,
        fillOpacity: info ? 0.82 : 0.25,
      };
    }

    function showDetail(feature) {
      var info = riesgo[ubigeo(feature)];
      var props = feature.properties;
      card.hidden = false;
      hint.hidden = true;
      elDepto.textContent = (props.FIRST_NOMB || "Perú");
      elProv.textContent = props.NOMBPROV || (info && info.provincia) || "—";
      if (info) {
        elNivel.textContent = info.nivel;
        elNivel.className = "nivel-badge nivel-" + info.nivel;
        elRN.textContent = Math.round(info.prob_rn * 100) + "%";
        elAD.textContent = info.pred_ad ? "brote" : "sin brote";
        elCasos.textContent = info.casos;
      } else {
        elNivel.textContent = "sin dato";
        elNivel.className = "nivel-badge";
        elRN.textContent = "—"; elAD.textContent = "—"; elCasos.textContent = "—";
      }
    }

    var layer = L.geoJSON(geo, {
      style: styleFor,
      onEachFeature: function (feature, lyr) {
        lyr.on({
          mouseover: function () { lyr.setStyle({ weight: 2.5, color: "#e8eef8" }); showDetail(feature); },
          mouseout: function () { layer.resetStyle(lyr); },
          click: function () { showDetail(feature); map.fitBounds(lyr.getBounds(), { maxZoom: 9, padding: [40, 40] }); },
        });
      },
    }).addTo(map);

    map.fitBounds(layer.getBounds(), { padding: [24, 24] });
  }).catch(function (err) {
    hint.textContent = "No se pudieron cargar los datos del mapa (" + err + ").";
  });
})();
