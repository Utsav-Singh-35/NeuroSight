# NeuraSight — Deployment Guide

## Deployment Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DEPLOYED ARCHITECTURE                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────┐                                     │
│  │  Vercel                     │                                     │
│  │  neurasight.vercel.app      │                                     │
│  │                             │                                     │
│  │  • React static build       │                                     │
│  │  • /api/* serverless fns    │──── HTTPS ────┐                     │
│  │  • Express logic as lambdas │               │                     │
│  └─────────────────────────────┘               │                     │
│                                                ▼                     │
│  ┌─────────────────────────────┐    ┌──────────────────────────┐    │
│  │  MongoDB Atlas              │    │  AWS EC2                  │    │
│  │  cluster0.l9w9qm.mongodb.net│    │  (FastAPI + 4 models)    │    │
│  │                             │    │                          │    │
│  │  • NeuraSight database      │    │  • t3.medium (4GB RAM)   │    │
│  │  • predictions collection   │    │  • Ubuntu 22.04          │    │
│  │  • projectdocs collection   │    │  • Docker + ensemble     │    │
│  │  • 512MB free storage       │    │  • Elastic IP            │    │
│  └─────────────────────────────┘    └──────────────────────────┘    │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  FUTURE: Oracle Cloud (Free Tier)                             │    │
│  │  • 24GB RAM, 4 CPUs, always-on VM — permanently free         │    │
│  │  • Migration target once AWS free tier expires                │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Platform Summary

| Service | Platform | URL | Status |
|---------|----------|-----|--------|
| **Frontend + API proxy** | Vercel | `https://neurasight.vercel.app` | ⏳ Pending |
| **ML Service (ensemble)** | AWS EC2 | `http://<elastic-ip>:8000` | ⏳ Pending |
| **Database** | MongoDB Atlas | `cluster0.l9w9qm.mongodb.net` | ✅ Active |
| **Future ML Service** | Oracle Cloud (free) | — | 🔮 Planned |

---

## 1. MongoDB Atlas (Database)

**Status:** ✅ Active

### Connection String
```
mongodb+srv://utsavsingh3529:****@cluster0.l9w9qm.mongodb.net/NeuraSight?retryWrites=true&w=majority
```

### Collections
| Collection | Purpose |
|-----------|---------|
| `predictions` | MRI classification history |
| `projectdocs` | Project documentation, architecture, datasets |

---

## 2. AWS EC2 (FastAPI ML Service)

**Status:** ⏳ Pending

### Recommended Instance

| Spec | Value | Why |
|------|-------|-----|
| **Instance type** | `t3.medium` | 4GB RAM — enough for ensemble (peak ~740MB per model, loaded sequentially) |
| **OS** | Ubuntu 22.04 LTS | Standard, Docker-friendly |
| **Storage** | 20GB EBS (gp3) | Model weights ~680MB + OS + Docker |
| **Elastic IP** | Yes | Static public IP for the FastAPI URL |
| **Security Group** | Port 8000 open (or 443 with nginx) | Allows Vercel to reach the ML service |

> **Cost:** t3.medium = ~$0.042/hr ≈ $30/mo. AWS free tier gives t2.micro (1GB RAM) for 12 months — NOT enough for the ensemble. You need t3.medium minimum.

### Step-by-Step Setup

#### Step 1: Launch EC2 Instance

1. Go to [AWS Console → EC2](https://console.aws.amazon.com/ec2/)
2. Click **Launch Instance**
3. Settings:
   - Name: `neurasight-ml`
   - AMI: Ubuntu 22.04 LTS (free tier eligible)
   - Instance type: `t3.medium` (4GB RAM, 2 vCPUs)
   - Key pair: Create new → download `.pem` file
   - Security group: Allow SSH (22), Custom TCP (8000)
   - Storage: 20GB gp3
4. Launch → note the public IP

#### Step 2: Allocate Elastic IP

1. EC2 → Elastic IPs → Allocate
2. Associate with your `neurasight-ml` instance
3. Note this IP — it won't change on restart

#### Step 3: SSH into the Instance

```bash
chmod 400 neurasight-ml.pem
ssh -i neurasight-ml.pem ubuntu@<ELASTIC-IP>
```

#### Step 4: Install Docker

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker ubuntu
# Log out and back in for group to take effect
exit
ssh -i neurasight-ml.pem ubuntu@<ELASTIC-IP>
```

#### Step 5: Create Project Directory

```bash
mkdir -p ~/neurasight/models
cd ~/neurasight
```

#### Step 6: Upload Model Files

From your local machine:
```bash
scp -i neurasight-ml.pem models/BRAIN_MRI_EFFICIENTNET.pth ubuntu@<ELASTIC-IP>:~/neurasight/models/
scp -i neurasight-ml.pem models/BRAIN_MRI_RESNET.pth ubuntu@<ELASTIC-IP>:~/neurasight/models/
scp -i neurasight-ml.pem models/BRAIN_MRI_DENSENET.pth ubuntu@<ELASTIC-IP>:~/neurasight/models/
scp -i neurasight-ml.pem models/BRAIN_MRI_VGG.pth ubuntu@<ELASTIC-IP>:~/neurasight/models/
scp -i neurasight-ml.pem models/meta_model.pkl ubuntu@<ELASTIC-IP>:~/neurasight/models/
scp -i neurasight-ml.pem models/ensemble_config.json ubuntu@<ELASTIC-IP>:~/neurasight/models/
```

#### Step 7: Upload FastAPI Code

```bash
scp -i neurasight-ml.pem -r backend/fastapi/app ubuntu@<ELASTIC-IP>:~/neurasight/
scp -i neurasight-ml.pem backend/fastapi/requirements.txt ubuntu@<ELASTIC-IP>:~/neurasight/
```

#### Step 8: Create Dockerfile on EC2

SSH into EC2 and create:
```bash
cat > ~/neurasight/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/

ENV MODELS_DIR=./models
ENV USE_ENSEMBLE=True
ENV MODEL_PATH=./models/BRAIN_MRI_EFFICIENTNET.pth
ENV CLASS_LABELS=Glioma,Meningioma,No Tumor,Pituitary
ENV GRADCAM_OPACITY=0.4
ENV GRADCAM_COLORMAP=jet
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
```

#### Step 9: Build and Run

```bash
cd ~/neurasight
docker build -t neurasight-ml .
docker run -d --name neurasight-ml --restart always -p 8000:8000 neurasight-ml
```

#### Step 10: Test

```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy","model_loaded":true}
```

From your local machine:
```bash
curl http://<ELASTIC-IP>:8000/health
```

---

## 3. Vercel (Frontend + Express Serverless Functions)

**Status:** ⏳ Pending

### Setup Steps

1. Push project to GitHub
2. Go to [vercel.com/new](https://vercel.com/new) → Import repo
3. Framework: Other (we have a custom `vercel.json`)
4. Root directory: `./`
5. Add environment variables in Vercel dashboard:

| Variable | Value |
|----------|-------|
| `FASTAPI_URL` | `http://<ELASTIC-IP>:8000` |
| `MONGODB_URI` | `mongodb+srv://utsavsingh3529:****@cluster0.l9w9qm.mongodb.net/NeuraSight?retryWrites=true&w=majority` |
| `FRONTEND_ORIGIN` | `https://neurasight.vercel.app` |
| `FASTAPI_TIMEOUT` | `60000` |

6. Deploy
7. Test: `https://neurasight.vercel.app/api/health`

---

## 4. Oracle Cloud (Future — Free Tier Migration)

**Status:** 🔮 Planned (after AWS free tier expires or for cost savings)

### Why Oracle Cloud?
- **24GB RAM, 4 CPUs, always-on VM — permanently free** (no expiry)
- Same Docker setup as AWS (just change the IP)
- No cold starts, no timeout limits

### Migration Steps (when ready)
1. Create Oracle Cloud account (free, needs credit card verification)
2. Launch "Always Free" ARM instance (Ampere A1, 24GB RAM)
3. Repeat Steps 4–10 from the AWS section above
4. Update Vercel env: `FASTAPI_URL` → new Oracle IP
5. Terminate AWS instance

---

## Environment Variables Reference

### Local Development (`python run.py`)
```env
# backend/fastapi/.env (ML service)
MODEL_PATH=../../models/BRAIN_MRI_EFFICIENTNET.pth
MODELS_DIR=../../models
USE_ENSEMBLE=False          ← False locally (8GB RAM can't handle VGG)
HOST=0.0.0.0
PORT=8000
CLASS_LABELS=Glioma,Meningioma,No Tumor,Pituitary
GRADCAM_OPACITY=0.4
GRADCAM_COLORMAP=jet

# backend/express/.env (loaded by api/_lib/config.js)
PORT=5000
MONGODB_URI=mongodb+srv://utsavsingh3529:****@cluster0.l9w9qm.mongodb.net/NeuraSight?retryWrites=true&w=majority
FASTAPI_URL=http://localhost:8000
FRONTEND_ORIGIN=http://localhost:3000
MAX_FILE_SIZE=10485760
FASTAPI_TIMEOUT=30000
HEALTH_CHECK_TIMEOUT=3000
```

### Deployed (AWS + Vercel)
```env
# AWS EC2 Docker container env vars
MODELS_DIR=./models
USE_ENSEMBLE=True           ← True on AWS (4GB RAM handles the ensemble)
MODEL_PATH=./models/BRAIN_MRI_EFFICIENTNET.pth
CLASS_LABELS=Glioma,Meningioma,No Tumor,Pituitary

# Vercel dashboard env vars
FASTAPI_URL=http://<ELASTIC-IP>:8000
MONGODB_URI=mongodb+srv://utsavsingh3529:****@cluster0.l9w9qm.mongodb.net/NeuraSight
FRONTEND_ORIGIN=https://neurasight.vercel.app
FASTAPI_TIMEOUT=60000
```

---

## Deployment Checklist

### Pre-deployment
- [x] MongoDB Atlas cluster created and accessible
- [x] Project docs uploaded to Atlas
- [ ] AWS account with EC2 access
- [ ] Vercel account created
- [ ] GitHub repo up to date

### AWS EC2 (ML Service)
- [ ] EC2 instance launched (t3.medium, Ubuntu 22.04)
- [ ] Elastic IP allocated and associated
- [ ] Docker installed
- [ ] Model weights uploaded (scp)
- [ ] FastAPI code uploaded
- [ ] Dockerfile created
- [ ] Container built and running
- [ ] `/health` endpoint responding from public IP
- [ ] Security group: port 8000 open

### Vercel (Frontend + API)
- [ ] Repo imported to Vercel
- [ ] Environment variables set (FASTAPI_URL, MONGODB_URI)
- [ ] Frontend builds successfully
- [ ] `/api/health` shows `fastapi_reachable: true`
- [ ] Full flow: upload → predict → gradcam → report

### Post-deployment
- [ ] CORS configured (FastAPI allows Vercel origin)
- [ ] Test from mobile device
- [ ] Error handling verified (EC2 down → graceful error)

---

## Cost Summary

| Service | Tier | Cost |
|---------|------|------|
| Vercel | Hobby (free) | $0 |
| AWS EC2 | t3.medium (on-demand) | ~$30/mo |
| MongoDB Atlas | M0 (free) | $0 |
| **Total** | | **~$30/mo** |

> **Cost reduction options:**
> - Use Spot Instance (save 60–70%): ~$10/mo
> - Migrate to Oracle Cloud free tier: $0/mo
> - Use t3.small (2GB): ~$15/mo — works if you drop VGG from ensemble

---

## Quick Reference Commands

### SSH into EC2
```bash
ssh -i neurasight-ml.pem ubuntu@<ELASTIC-IP>
```

### Check container status
```bash
docker ps
docker logs neurasight-ml --tail 50
```

### Restart container
```bash
docker restart neurasight-ml
```

### Update code on EC2
```bash
# From local machine
scp -i neurasight-ml.pem -r backend/fastapi/app ubuntu@<ELASTIC-IP>:~/neurasight/
ssh -i neurasight-ml.pem ubuntu@<ELASTIC-IP> "cd ~/neurasight && docker build -t neurasight-ml . && docker stop neurasight-ml && docker rm neurasight-ml && docker run -d --name neurasight-ml --restart always -p 8000:8000 neurasight-ml"
```

---

**Last Updated:** August 2026
