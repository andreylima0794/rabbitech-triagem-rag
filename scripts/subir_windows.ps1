# Confere Docker Desktop (WSL2) e GPU NVIDIA, cria o .env e sobe o compose
# (Ollama com GPU + Chroma), baixando os modelos na primeira vez.
# Rodar na raiz do projeto:
#   powershell -ExecutionPolicy Bypass -File .\scripts\subir_windows.ps1
# O log vai para logs\subir_windows.log (fora do git: pode ter dados da maquina).

[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:WSL_UTF8 = "1"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs\subir_windows.log"
$utf8 = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText((Join-Path $PWD $log), "", $utf8)

# no PowerShell 5.1, Tee-Object/Set-Content gravam UTF-16/ANSI; aqui o log sai sempre em UTF-8
function Registra([string]$texto) {
    Write-Host $texto
    [IO.File]::AppendAllText((Join-Path $PWD $log), $texto + "`r`n", $utf8)
}

function Passo($titulo, [scriptblock]$cmd) {
    Registra "`n=== $titulo"
    Registra (& $cmd 2>&1 | Out-String -Width 200)
}

Registra ("inicio: " + (Get-Date -Format s))

Passo "WSL" { wsl --status; wsl -l -v }
Passo "Driver NVIDIA no Windows" { nvidia-smi }
Passo "Docker" { docker version; docker info --format "SO: {{.OperatingSystem}} | servidor: {{.ServerVersion}} | runtimes: {{json .Runtimes}}" }
Passo "Docker Compose" { docker compose version }

# sem o engine do Docker Desktop nada do que vem depois funciona: tenta abrir e espera ate 3 min
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    $exe = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $exe)) { Registra "`nDocker Desktop nao encontrado em $exe. Instale e rode de novo."; exit 1 }
    Registra "`nEngine do Docker parado: abrindo o Docker Desktop e esperando..."
    Start-Process $exe
    $limite = (Get-Date).AddMinutes(3)
    do { Start-Sleep 5; docker info *> $null } while ($LASTEXITCODE -ne 0 -and (Get-Date) -lt $limite)
    if ($LASTEXITCODE -ne 0) { Registra "Engine nao respondeu em 3 min. Abra o Docker Desktop manualmente e rode de novo."; exit 1 }
    Passo "Docker (engine no ar)" { docker info --format "SO: {{.OperatingSystem}} | servidor: {{.ServerVersion}} | runtimes: {{json .Runtimes}}" }
}

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Registra "`n.env criado a partir do .env.example"
}

Passo "Subindo os servicos" { docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d }

# o download aparece aqui no console; no log fica so o final
Write-Host "`n=== Baixando os modelos (pode levar varios minutos)..."
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f modelos

Passo "Fim do download" { docker compose logs --no-log-prefix --tail 5 modelos }
Passo "Estado dos servicos" { docker compose -f docker-compose.yml -f docker-compose.gpu.yml ps -a }
Passo "GPU vista pelo Ollama" { docker compose logs ollama | Select-String -Pattern "inference compute|no compatible GPUs" }
Passo "Modelos instalados" { docker compose exec ollama ollama list }
Passo "Ollama /api/version" { Invoke-RestMethod http://127.0.0.1:11434/api/version | ConvertTo-Json -Compress }
Passo "Chroma /api/v2/heartbeat" { Invoke-RestMethod http://127.0.0.1:8000/api/v2/heartbeat | ConvertTo-Json -Compress }

Registra ("`nfim: " + (Get-Date -Format s))
