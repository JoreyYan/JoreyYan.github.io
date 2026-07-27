const slugify = (text, index) =>
  `${text.trim().replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-|-$/g, "") || "section"}-${index}`;

fetch("article.md")
  .then((response) => {
    if (!response.ok) throw new Error("文章载入失败");
    return response.text();
  })
  .then((markdown) => {
    const article = document.querySelector("#article");
    article.innerHTML = marked.parse(markdown);

    const headings = [...article.querySelectorAll("h2")];
    const toc = document.querySelector("#toc");
    headings.forEach((heading, index) => {
      heading.id = slugify(heading.textContent, index);
      const link = document.createElement("a");
      link.href = `#${heading.id}`;
      link.textContent = heading.textContent;
      toc.appendChild(link);
    });

    if (window.renderMathInElement) {
      renderMathInElement(article, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false }
        ],
        throwOnError: false
      });
    }
  })
  .catch(() => {
    document.querySelector("#article").innerHTML =
      "<p>文章暂时无法载入，请稍后刷新页面。</p>";
  });
