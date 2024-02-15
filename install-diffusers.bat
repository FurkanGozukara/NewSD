@echo off
SET REPOSITORIES_DIR="repositories"

if not EXIST .\%REPOSITORIES_DIR% mkdir %REPOSITORIES_DIR%
echo Cloning Diffusers repository...
cd .\repositories
git clone https://github.com/kashif/diffusers.git
cd diffusers
git checkout a3dc21385b7386beb3dab3a9845962ede6765887 2>nul
echo Patching Diffusers...
copy /y ..\..\patch\Diffusers\modeling_wuerstchen_diffnext.py .\src\diffusers\pipelines\wuerstchen
echo Installing Diffusers...
pip install -e .