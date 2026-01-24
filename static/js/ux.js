document.addEventListener("DOMContentLoaded", () => {
  const c = document.querySelector(".container");
  if (c){
    c.style.opacity = 0;
    setTimeout(()=>{ c.style.transition="opacity .4s ease"; c.style.opacity=1; }, 60);
  }
});
// Bootstrap tooltips
document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => new bootstrap.Tooltip(el));
