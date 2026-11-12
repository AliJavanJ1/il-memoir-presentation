# Imitation by Occupancy Matching and Forward–Backward Representations (Presentation)

Interactive [Slidev](https://sli.dev/) presentation for the Master's memoir:

> **Imitation by occupancy matching and by forward–backward representations: Proofs, approximations and a controlled comparison of f-PG and FB imitation**  
> *Ali Javani — supervised by Olivier Cappé (October 2026)*

## Overview

This presentation covers the theoretical analysis, algorithmic approximations, and controlled empirical comparison of two state-only imitation learning families (without reward signals or expert actions):

1. **Direct $f$-divergence occupancy matching**:
   - **$f$-IRL** (Ni et al., 2020) — studied in theory
   - **$f$-PG** (Agarwal et al., 2023) — online policy gradient matching the expert's state occupancy
2. **Forward–backward (FB) representations**:
   - **FB pretraining** (Touati & Ollivier, 2021) — reward-free world model learned offline from task-blind replay
   - **ER-FB & DM-FB** (Pirotta et al., 2024) — zero-interaction offline imitation in a low-dimensional latent space $z$

The slides walk through 5 core questions:
- **Q1**: Unified notation, core proofs, and exact vs. approximate steps
- **Q2**: Method comparison and practical use cases
- **Q3**: Key hyperparameters ($f$-divergence choice, representation rank $d$, orthonormality penalty)
- **Q4**: Exact vs. learned behavior in finite state spaces (Four Rooms maze)
- **Q5**: Scaling to continuous state spaces with neural networks

---

## Live Presentation (GitHub Pages)

Once deployed via GitHub Actions, the interactive slides are available at:

`https://<your-github-username>.github.io/<repository-name>/`

*(Press `Space` / arrow keys to navigate, `o` for slide overview, `p` for presenter mode with speaker notes, and `d` to toggle dark mode.)*

---

## Running Locally

### Prerequisites

- [Bun](https://bun.sh/) (v1.x) — or [Node.js](https://nodejs.org/) (v20+)
-*(Optional)* [mise](https://mise.jdx.dev/) (`mise install` will automatically set up Bun and Node from `mise.toml`)

### 1. Install Dependencies

```bash
bun install
```

*(Or with npm: `npm install`)*

### 2. Start the Local Dev Server

```bash
bun run dev
```

*(Or `npx slidev`)*

Then open [http://localhost:3030](http://localhost:3030) in your browser. Changes to `slides.md` hot-reload automatically.

### 3. Build Static Site

To build the static Single-Page Application into `dist/`:

```bash
bun run build
```

---

## Project Structure

| File / Folder | Description |
|---|---|
| `slides.md` | Main Slidev deck (Markdown, LaTeX math, Vue components, and speaker notes) |
| `public/` | Static figures and heatmaps generated from the memoir experiments |
| `slide-bottom.vue` | Custom footer displaying current / total slide numbers |
| `.github/workflows/deploy.yml` | GitHub Actions workflow that builds and deploys the slides to GitHub Pages |
| `package.json` | Slidev dependencies and scripts (`dev`, `build`, `export`) |
