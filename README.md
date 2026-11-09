# How Much of Your VLA is kNN: project page

Static project page for the CoRL 2026 workshop paper, served by GitHub Pages from the root of this `gh-pages` branch.
It is based on the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)
and follows the layout of the [BorealTC page](https://github.com/norlab-ulaval/BorealTC/tree/gh-pages).

## Preview locally

```bash
python3 -m http.server 8765
# open http://127.0.0.1:8765
```

## Publish

Push this branch, then in the repository settings select **Pages → Deploy from a branch → `gh-pages` / root**.
The page will be served at `https://sereact.github.io/lerobot_knn/`.

## Contents

- `index.html`: the page. Remaining `TODO` comments: workshop name, author links, arXiv button.
- `static/images/`: figures cropped at 300 dpi from the final paper PDF, plus the 1200×630 social preview.
- `static/js/index.js`: the results chart (data from Table 1 of the paper), suite tabs, and video autoplay on scroll.
- `static/videos/rollouts/`: one successful LIBERO episode per task (`<suite>_<task>.mp4`) from the multimodal
  k-NN evaluation run (91.7 %), re-encoded at 2× speed with poster frames.
- `static/pdfs/how_much_of_your_vla_is_knn.pdf`: the final paper.
