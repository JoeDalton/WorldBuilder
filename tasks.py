import platform
import shutil

from invoke import task

@task
def check_dynamic_versioning(c):
    result = c.run("poetry self show plugins", hide=True, warn=True)
    if "poetry-dynamic-versioning" not in result.stdout:
        print(
            "poetry-dynamic-versioning plugin not found, installing it "
            "(required for poetry install/build to pick up the git-derived version)..."
        )
        c.run("poetry self add poetry-dynamic-versioning[plugin]")


@task
def native(c):
    """Build the native terrain collision library (docs/source/native_build.md)."""
    if platform.system() == "Windows":
        c.run("poetry run powershell -NoProfile -File scripts/build_native.ps1")
    else:
        c.run("poetry run scripts/build_native.sh")


@task(pre=[check_dynamic_versioning])
def develop(c):
    c.run("poetry install")
    c.run("poetry run pre-commit install")
    # Without the library, terrain collision raises: only warn, as the build
    # needs a C++ compiler a developer may not have.
    if not c.run("invoke native", warn=True).ok:
        print(
            "warning: the native terrain collision library did not build, "
            "Terrain.enable_collision will raise. Retry with `invoke native`."
        )


@task
def test(c):
    c.run("poetry run pytest tests/ -n auto")


@task()
def install(c):
    c.run("poetry install --without dev,test")


@task
def quality(c):
    c.run("poetry run pre-commit run --all-files")


@task
def check_env(c):
    c.run("pip freeze |grep numpy ; pip freeze |grep scipy ; pip freeze |grep world_builder")


@task
def deploy(c):
    c.run("poetry publish --build --repository world_builder")


@task
def coverage(c):
    c.run(
        "poetry run pytest tests -v -n auto --cov-report term --cov-report html:htmlcov --cov-report xml --cov=./src/"
    )
    c.run("poetry run coverage report")
    c.run("poetry run coverage xml")


@task
def doc(c, path=r"./docs/build"):
    c.run("poetry install --with docs")
    c.run(f'poetry run sphinx-build -M html docs/source "{path}/latest"')
    print(f"Doc written in {path}/latest")
    c.run(f'poetry run sphinx-build -M html docs/source "{path}/$(poetry version -s)"')
    print(f"Doc written in {path}/$(poetry version -s)")


@task
def clean(c):
    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("dist", ignore_errors=True)
    shutil.rmtree("iframe_figures", ignore_errors=True)
    shutil.rmtree("target", ignore_errors=True)