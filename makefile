# eqmagene makefile

PYTHON=python

help:
    @echo "Available tasks: "
    @echo "help     --> Thist help page"
    @echo "test     --> Run unit tests"
    @echo "flake8   --> Check pep8 errors"
    @echo "sdist    --> Make source distribution"
    @echo "pypi     --> Upload to pypi"
    @echo "coverage --> html report for unit test results"
    @echo "clean    --> Remove all the build files for fresh start"

 test:
    ${PYTHON} -c "import eqmagene; eqmagene.test()"

 flake8:
        flake8 .

 sdist: clean
        ${PYTHON} setup.py sdist
        git status

 coverage:
	rm -rf cover
	nosetests --with-coverage --cover-html --cover-package=eqmagene .
	firefox cover/index.html

clean:
	rm -f MANIFEST
	rm -rf cover
	rm -rf *.lprof
	rm -rf *.egg-info
	rm -rf build dist some_sums.egg-info
	find . -name \*.pyc -delete
	rm -rf build
