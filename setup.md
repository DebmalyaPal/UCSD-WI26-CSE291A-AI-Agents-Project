# CREW-WILDFIRE REPLICATION SETUP GUIDE

This guide explains how to replicate and run the CREW Wildfire simulation
from the General Robotics Lab CREW repository across:

- macOS (Intel + Apple Silicon)
- Windows 10/11
- Linux (Ubuntu recommended)

The architecture consists of:

1) Unity Simulation (environment + physics)
2) Go/Nakama server (networking + state relay)
3) Python Agents (LLM + planning logic, usually in Docker)
4) Optional: LLM API (OpenAI or other provider)

------------------------------------------------------------
SECTION 1 — SYSTEM REQUIREMENTS
------------------------------------------------------------

Minimum:
- 16 GB RAM
- 20 GB free disk
- Docker Desktop
- Go 1.18+
- Unity Hub (2021 LTS or newer)
- Python 3.10+

Recommended:
- 32 GB RAM
- SSD
- Dedicated GPU (not required but helpful)

------------------------------------------------------------
SECTION 2 — INSTALL DEPENDENCIES
------------------------------------------------------------

MACOS (Intel or Apple Silicon)
------------------------------------------------------------

1) Install Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2) Install Go [OPTIONAL]
```bash
brew install go
```

3) Install Docker Desktop (Apple Silicon version if M-series). 
https://www.docker.com/products/docker-desktop

4) Install Unity Hub.  
https://unity.com/download

   Inside Unity Hub:
   - Install Unity 2021 LTS (or version specified by repo)
   - Add macOS Build Support

5) Verify architecture
```bash
uname -m
```
Expected for M-series: arm64

WINDOWS
------------------------------------------------------------

1) Install WSL2 (recommended)
```bash
wsl --install
```

2) Install Docker Desktop (enable WSL integration).  
https://www.docker.com/products/docker-desktop

3) Install Go (Windows installer).  [OPTIONAL]
https://go.dev/dl/

4) Install Unity Hub.  
https://unity.com/download
   - Install Unity 2021 LTS
   - Add Windows Build Support

LINUX (Ubuntu)
------------------------------------------------------------

1) Install Docker
```bash
sudo apt update
sudo apt install docker.io
sudo systemctl enable docker
sudo systemctl start docker
```

2) Install Go [OPTIONAL]
```bash
sudo snap install go --classic
```

3) Install Unity Hub.  
https://docs.unity3d.com/hub/manual/InstallHub.html

------------------------------------------------------------
SECTION 3 — CLONE CREW REPOSITORY
------------------------------------------------------------
```bash
git clone https://github.com/generalroboticslab/CREW.git
cd CREW
```

------------------------------------------------------------
SECTION 4 — START NAKAMA (SERVER LAYER)
------------------------------------------------------------

CREW uses Nakama for networking.  

From project root:  
```bash
cd crew-dojo
docker compose up -d

# OR

docker compose -f docker-compose.nakama.yml up -d
```

Verify server running: `http://localhost:7350`.  

If successful, Nakama server is active.

------------------------------------------------------------
SECTION 5 — SETUP PYTHON AGENT ENVIRONMENT
------------------------------------------------------------

**Option A — Using Conda (Recommended)**

```bash
conda create -n crew python=3.10
conda activate crew


cd crew-algorithms/crew_algorithms/wildfire_alg

pip install -r requirements.txt
# OR if poetry is used:
poetry install
```

Set your LLM API key:  

```bash
export OPENAI_API_KEY=your_key_here      (macOS/Linux)
setx OPENAI_API_KEY your_key_here        (Windows)
```
  
------------------------------------------------------------
  
**Option B — Using Python venv (Lightweight)**

1) Ensure Python 3.10 is installed

Mac:
```bash
brew install python@3.10
```

Linux:
```bash
sudo apt install python3.10 python3.10-venv
```

Windows:  
Install Python 3.10 from python.org

2) Create virtual environment

```bash
python3.10 -m venv crew-env
```

3) Activate environment

Mac/Linux:
```bash
source crew-env/bin/activate
```

Windows:
```bash
crew-env\Scripts\activate
```

4) Install dependencies

```bash
cd crew-algorithms/crew_algorithms/wildfire_alg
pip install --upgrade pip
pip install -r requirements.txt

# OR if poetry is used in repo
pip install poetry
poetry install
```

5) Set API key

Mac/Linux:
```bash
export OPENAI_API_KEY=your_key_here
```

Windows:
```bash
setx OPENAI_API_KEY your_key_here
```

------------------------------------------------------------

**Option C — Docker-Only Agents (Most Reproducible for Teams)**

Skip local Python entirely and run agents inside Docker.

Example:
```bash
docker build -t crew-agents .
docker run --env OPENAI_API_KEY=your_key_here crew-agents
```

------------------------------------------------------------
SECTION 6 — BUILD AND RUN UNITY SIMULATION
------------------------------------------------------------

1) Open Unity Hub
2) Add project: `CREW/crew-simulation` (or path defined in repo)
3) Open project
4) Press Play to test
   OR
5) Build headless version:  
   File → Build Settings
   - Select correct platform
   - Enable "Server Build" if available
   - Build
  
Run built executable:
  
Mac/Linux: `./WildfireSimulation.x86_64`
Windows: `WildfireSimulation.exe`

The simulation will connect to Nakama automatically.

------------------------------------------------------------
SECTION 7 — RUN AGENTS
------------------------------------------------------------

From wildfire_alg directory:

```bash
python run_experiment.py --algorithm camon
```

Available algorithms:
- camon
- coela
- embodied
- hmas2

Agents will:
1) Connect to Nakama server
2) Receive environment state
3) Send actions back to simulation

------------------------------------------------------------
SECTION 8 — NETWORKING ARCHITECTURE
------------------------------------------------------------

Unity Simulation <-> Nakama (Go server).  
Nakama (Go server) <-> Python Agents (WebSocket / RPC).  

> REQUIRED STARTUP ORDER
> 1. Start Docker Desktop
> 2. Start Nakama (docker compose up -d)
> 3. Start Unity simulation
> 4. Start Python agents
>
> If order is wrong:
> - Agents will fail to register
> - Unity will fail to attach to server

On macOS Docker:  
Agents must connect to: `host.docker.internal:7350`

On Linux:  
Use: `localhost:7350`

On Windows:  
Use: `localhost:7350` OR WSL IP if needed

------------------------------------------------------------
SECTION 9 — VERIFY FULL PIPELINE
------------------------------------------------------------

Checklist:

[ ] Docker running.  
[ ] Nakama server running (port 7350).  
[ ] Unity simulation running.  
[ ] Python agents connected.  
[ ] Logs show state updates.  
[ ] Agents moving in simulation.  

------------------------------------------------------------
SECTION 10 — SCALING NOTES
------------------------------------------------------------

For >100 agents:
- Use headless Unity
- Increase Docker memory limit
- Consider cloud VM

------------------------------------------------------------
SECTION 11 — CLEAN SHUTDOWN
------------------------------------------------------------

Stop agents:
```bash
CTRL+C
```

Stop Unity:
Close window

Stop Nakama:
```bash
docker compose down
```
