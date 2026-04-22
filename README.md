# CUDA Auto-Execution Pipeline

Write a `.cu` file, `git push`, and see GPU results as a commit comment in ~60 seconds.

```
git push  →  GitHub Action  →  Colab GPU  →  commit comment
```

---

## How It Works

```
┌─────────────┐   push kernels/*.cu   ┌──────────────────┐
│  Your Mac   │ ───────────────────►  │  GitHub Action   │
└─────────────┘                       └────────┬─────────┘
                                               │ writes jobs/current.json
                                               ▼
                                      ┌──────────────────┐
                                      │  GitHub Repo     │◄──── polls every 10s
                                      └────────┬─────────┘             │
                                               │ job available          │
                                               ▼                        │
                                      ┌──────────────────┐             │
                                      │  Colab GPU       │ ────────────┘
                                      │  (T4 / A100)     │
                                      └────────┬─────────┘
                                               │ writes results/{sha}.txt
                                               ▼
                                      ┌──────────────────┐
                                      │  GitHub Action   │  posts commit comment
                                      └──────────────────┘
```

The Colab notebook runs as a **persistent GPU worker** — start it once and it handles all
subsequent pushes until the Colab session times out (~12 h free tier).

---

## Project Structure

```
cuda-kernels/
├── .github/
│   └── workflows/
│       └── run_cuda.yml       # GitHub Action: detect push → submit job → post comment
├── kernels/
│   └── matmul_naive.cu        # Your CUDA kernels go here
├── jobs/
│   └── current.json           # Job queue (written by Action, read by Colab)
├── results/
│   └── {sha}.txt              # Results (written by Colab, read by Action)
├── auto_runner.ipynb          # Colab notebook — the GPU worker
├── trigger_colab.py           # Called by the Action: submits job & polls results
├── requirements.txt
└── setup.sh
```

---

## One-Time Setup

### 1. Run the setup script

```bash
chmod +x setup.sh
./setup.sh
```

### 2. Create a GitHub repo and push

```bash
git init
git remote add origin https://github.com/YOUR_USER/cuda-kernels.git
git add .
git commit -m "Initial commit"
git push -u origin main
```

### 3. Add GitHub Secrets

Go to **Settings → Secrets → Actions** in your repo and add:

| Secret | Value |
|--------|-------|
| `GH_PAT` | Personal Access Token with **`repo`** scope ([create here](https://github.com/settings/tokens)) |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | *(optional)* Service account JSON for Drive notebook annotation |
| `NOTEBOOK_ID` | *(optional)* Google Drive file ID of `auto_runner.ipynb` |

> `GH_PAT` is required. The Google secrets are optional — they enable annotating the
> Colab notebook in Drive with the active job, but the pipeline works without them.

### 4. Start the Colab GPU worker

1. Upload `auto_runner.ipynb` to [Google Drive](https://drive.google.com)
2. Open it in [Google Colab](https://colab.research.google.com)
3. **Runtime → Change runtime type → GPU (T4)**
4. Open the **🔑 Secrets** sidebar and add:
   - `GH_PAT` — same token as above
   - `GITHUB_REPO` — `your-username/cuda-kernels`
5. **Runtime → Run all**

The notebook prints `Waiting for jobs...` and polls every 10 seconds.

---

## Daily Usage

```bash
# Write or edit a kernel
vim kernels/my_kernel.cu

# Push — that's it
git add kernels/my_kernel.cu
git commit -m "test new kernel"
git push
```

Open the commit on GitHub. Within ~60 seconds you'll see a comment like:

```
✅ CUDA Kernel Results

Kernel: kernels/matmul_naive.cu
Commit: a1b2c3d4

GPU:       Tesla T4, 15360 MiB
Kernel:    kernels/matmul_naive.cu
Commit:    a1b2c3d4

GPU:        Tesla T4
Matrix:     1024x1024
Time:       2.347 ms
GFLOPS:     916.42
C[0][0]:    255.982300
C[N/2][N/2]: 256.012543
```

---

## Adding New Kernels

1. Create `kernels/your_kernel.cu`
2. Make sure it has a `main()` that prints results to stdout
3. Push — the pipeline handles the rest

```c
// kernels/vector_add.cu
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    // ... your code ...
    printf("Result: %.6f\n", result);
    return 0;
}
```

---

## Secrets Reference

| Secret | Required | Description |
|--------|----------|-------------|
| `GH_PAT` | **Yes** | GitHub PAT (repo scope) — lets Action push job files and Colab push results |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | No | GCP service account JSON — annotates Drive notebook with active job info |
| `NOTEBOOK_ID` | No | Google Drive file ID of `auto_runner.ipynb` |

---

## Troubleshooting

**Action posts "timed out"**
- Check the Colab notebook is still running (free tier dies after ~12 h of inactivity)
- Restart it: Runtime → Run all

**`nvcc` compilation fails**
- The error will appear in the commit comment — fix the `.cu` file and push again

**`GH_PAT` permission denied**
- Make sure the token has `repo` (not just `public_repo`) scope
- Regenerate at https://github.com/settings/tokens

**Multiple pushes at once**
- Only one job runs at a time; the Colab notebook processes them sequentially
- Each commit gets its own results file keyed by SHA

**Infinite workflow loops**
- All Colab commits use `[ci skip]` in the message, which the workflow ignores
- The workflow also only triggers on `kernels/**.cu` changes, not on `jobs/` or `results/`
