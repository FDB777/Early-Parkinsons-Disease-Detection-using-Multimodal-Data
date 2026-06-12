function showName(input, labelId) {
  const el = document.getElementById(labelId);
  el.textContent = input.files[0] ? "OK " + input.files[0].name : "";
}

document.querySelectorAll(".file-drop").forEach((drop) => {
  drop.addEventListener("dragover", () => drop.classList.add("dragover"));
  drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
  drop.addEventListener("drop", () => drop.classList.remove("dragover"));
});
