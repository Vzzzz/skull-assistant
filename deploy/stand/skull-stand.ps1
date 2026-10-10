<#
.SYNOPSIS
  Запуск стенда Skull Assistant в WSL в обычном или сервисном режиме.

.DESCRIPTION
  Сервисный режим = флаг-файл $env:USERPROFILE\.skull-maintenance. Юниты skull-* и ollama
  имеют ConditionPathExists=!/mnt/c/Users/valer/.skull-maintenance и при наличии флага не
  стартуют вместе с WSL. Флаг ставится/снимается при выключенном WSL, гонки нет.

.EXAMPLE
  .\skull-stand.ps1 service     # флаг + запуск WSL без сервисов (скачать модели, обновить пакеты)
  .\skull-stand.ps1 normal      # снять флаг + запуск WSL, сервисы стартуют
  .\skull-stand.ps1 status      # состояние флага, WSL и юнитов
  .\skull-stand.ps1 stop        # остановить WSL (и все сервисы)
#>
param(
  [ValidateSet('normal','service','status','stop')]
  [string]$Mode = 'status',
  [string]$Distro = 'Ubuntu-22.04'
)
$flag = Join-Path $env:USERPROFILE '.skull-maintenance'

switch ($Mode) {
  'service' {
    New-Item -ItemType File -Path $flag -Force | Out-Null
    Write-Host "[service] флаг установлен: $flag — skull-* и ollama не стартуют"
    # если WSL уже запущен, гасим то, что успело подняться (ollama — только с sudo, напоминаем)
    wsl -d $Distro -- bash -lc 'systemctl --user stop skull.target 2>/dev/null; systemctl is-active ollama >/dev/null 2>&1 && echo "ollama активен: sudo systemctl stop ollama" || true'
    wsl -d $Distro --cd ~/skull-assistant
  }
  'normal' {
    if (Test-Path $flag) { Remove-Item $flag -Force; Write-Host "[normal] флаг снят" }
    wsl -d $Distro -- bash -lc 'systemctl --user start skull.target 2>/dev/null; systemctl is-active ollama >/dev/null 2>&1 || echo "ollama не активен: sudo systemctl start ollama"'
    wsl -d $Distro --cd ~/skull-assistant
  }
  'status' {
    Write-Host ("флаг сервисного режима: " + $(if (Test-Path $flag) { 'ЕСТЬ (сервисы не стартуют)' } else { 'нет' }))
    wsl -l -v | Out-String | Write-Host
    wsl -d $Distro -- bash -lc 'systemctl --user list-units "skull*" --no-pager 2>/dev/null; echo "ollama: $(systemctl is-active ollama 2>/dev/null)"; ollama ps 2>/dev/null'
  }
  'stop' {
    wsl -t $Distro
    Write-Host "[stop] $Distro остановлен"
  }
}
