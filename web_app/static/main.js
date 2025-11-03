const thumbs = document.getElementById("thumbs");
const lastRefresh = document.getElementById("lastRefresh");

async function fetchLatest() {
  try {
    const res = await fetch("/api/latest_images", { cache: "no-store" });
    const data = await res.json();

    thumbs.innerHTML = "";

    if (!Array.isArray(data) || data.length === 0) {
      thumbs.innerHTML = `<div class="text-secondary small">Sin capturas aún.</div>`;
      return;
    }

    data.forEach((item) => {
      const card = document.createElement("div");
      card.innerHTML = `
        <img class="thumb" src="${item.url}" alt="${item.name}" />
        <div class="mt-1 d-flex justify-content-between align-items-center">
          <span class="text-truncate small">${item.name}</span>
          <a class="small" href="${item.url}" download>Descargar</a>
        </div>
      `;
      thumbs.appendChild(card);
    });

    const dt = new Date();
    lastRefresh.textContent = `Actualizado: ${dt.toLocaleTimeString()}`;
  } catch (e) {
    console.error(e);
  }
}

// Primer fetch inmediato y luego cada 2.5s
fetchLatest();
setInterval(fetchLatest, 2500);
