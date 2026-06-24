const fadeObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
      }
    });
  },
  { threshold: 0.12 },
);

document.querySelectorAll(".fade-up").forEach((node) => fadeObserver.observe(node));

const copyButton = document.querySelector(".copy-btn");
const bibtex = document.querySelector(".bibtex");

copyButton?.addEventListener("click", async () => {
  const text = bibtex?.innerText.trim();
  if (!text) return;

  try {
    await navigator.clipboard.writeText(text);
    copyButton.textContent = "Copied";
    copyButton.classList.add("copied");
    setTimeout(() => {
      copyButton.textContent = "Copy";
      copyButton.classList.remove("copied");
    }, 1400);
  } catch {
    copyButton.textContent = "Copy failed";
  }
});
