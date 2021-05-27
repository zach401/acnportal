# Contributing to acnportal

## Reporting issues

When reporting issues please include as much detail as possible about your
operating system, acnportal version and python version, along with versions of dependencies.
Whenever possible, please
also include a brief, self-contained code example that demonstrates the problem.

## Requesting features

Please use the issue tracker on github to request features.

In your request, please include the following:

**Is your feature request related to a problem? Please describe.**
(A clear and concise description of what the problem is. Ex. I'm always frustrated when \[...])

**Describe the solution you'd like.**
(A clear and concise description of what you want to happen)

**Describe alternatives you've considered.**
(A clear and concise description of any alternative solutions or features you've considered.)

**Additional context**
(Add any other context or screenshots about the feature request here.)

Also, if you want to propose an implementation for a solution, feel free to include it in the issue. Also, please reference related issues and bugs if you know of any.

## Contributing code

Thanks for your interest in contributing to acnportal!

We use a development process similar to that of [NumPy](<https://numpy.org/devdocs/dev/index.html>). As such, some of this document is copied from the aforementioned link.

1.  If you are a first-time contributor:

    -   Go to <https://github.com/zach401/acnportal> and click the “fork” button to create your own copy of the project.
        Clone the project to your local computer:
    
        ```bash
        git clone https://github.com/your-username/acnportal.git
        ```

    -   Change the directory:
    
        ```bash
        cd acnportal
        ```

    -   Add the upstream repository:
    
        ```bash
        git remote add upstream https://github.com/zach401/acnportal.git
        ```

    -   Now, `git remote -v` will show two remote repositories named:
    
        `upstream`, which refers to the acnportal repository
        `origin`, which refers to your personal fork

2.  Develop your contribution:

    -   Pull the latest changes from upstream:
    
        ```bash
        git checkout master
        git pull upstream master
        ```

    -   Create a branch for the feature you want to work on. Since the branch name will appear in the merge message, use a sensible name. Here
        we'll use `is-feasible-speedups`.
    
        ```bash
        git checkout -b is-feasible-speedups
        ```

    -   Commit locally as you progress (git add and git commit). Use a properly formatted commit message (54 char title, 72 char/line  description preferred). **Write tests that fail before your change and pass afterward**, run all the tests locally. Be sure to document any changed behavior in docstrings, keeping to the [Google docstring standard](<https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>).

3.  To submit your contribution:

    -   Push your changes back to your fork on GitHub:
    
        ```bash
        git push origin is-feasible-speedups
        ```

    -   Enter your GitHub username and password (repeat contributors or advanced users can remove this step by connecting to GitHub with SSH).
    
    -   Go to GitHub. The new branch will show up with a green Pull Request button. **Please merge into the `dev` branch of acnportal**! If accepted, your changes will be integrated on the next package release. Make sure the title and message are clear, concise, and self-explanatory. Then click the button to submit it.
    
    -   If your PR cannot be merged automatically, please resolve merge conflicts on your development branch and resubmit the PR.

4.  Review process:

    -   Reviewers (the other developers and interested community members) will write inline and/or general comments on your Pull Request (PR) to help you improve its implementation, documentation and style. We're all very friendly and open to learning from each other while improving the code, so don't let the review process discourage you from contributing! Also, if your PR is not getting any reaction, feel free to email the developer team at <mailto:ev-help@caltech.edu>.
    -   To update your PR, make your changes on your local repository, commit, run tests, and **only if they succeed** push to your fork. As soon as those changes are pushed up (to the same branch as before) the PR will update automatically. If you have no idea how to fix the test failures, you may push your changes anyway and ask for help in a PR comment.
    -   Various continuous integration (CI) services are triggered after each PR update to build the code, run unit tests, and check coding style of your branch. The CI tests should pass before your PR can be merged. If CI fails, you can find out why by clicking on the “failed” icon (red cross) and inspecting the build and test log. To avoid overuse and waste of this resource, test your work locally before committing. If you can't figure out why one or more steps of the CI pipeline is failing, you can ask for help in a PR comment.
    -   A PR must be approved by at least one core team member before merging. Approval means the core team member has carefully reviewed the changes, and the PR is ready for merging. A member of the team will give the LGTM and you may then merge if a member of the team has not already done so.

5.  User-facing changes: when a PR makes user-facing changes, we have some general guidelines we like to follow:

    -   Avoid changing user-facing function signatures/names. If absolutely necessary, add arguments using kwargs with defaults; "remove" arguments by deprecating the argument when input. If changing a function or attribute name, add a deprecated access to the old function/attribute name.
    -   You may need to add additional files/modules to the docs so that readthedocs finds the new docstrings.
    -   New dependencies should be avoided if possible. Depending on the importance of the feature and the activity of the new dependency, the core team may overrule this, but such should be discussed on Github.
    -   If you are proposing a change that affects multiple packages of the ACN Research Portal, please contact the developer team at <mailto:ev-help@caltech.edu> to discuss.

## Stylistic Guidelines

We want to follow PEP 8, so please set your editor as such and use flake8 or pyflakes or similar to lint.
We use [black](<https://github.com/psf/black>) with default settings to auto format our
code. Please run black before submitting a pull request.

Also, we're currently trying to type hint our code more thoroughly so that we may include a Mypy build in CI. Please type hint your additional code.

We know that in its current form, the code does not completely follow PEP 8. We're working on conforming to PEP 8, type hinting code, and including more tests, so if you want an easy introduction to the package, feel free to help out with that!

Some additional miscellaneous style guidelines:

-   Closing parens for hanging indents should occupy a new line.

-   For classes, place docstrings in the class declaration rather than in `__init__`, unless `__init__` does significant processing on its inputs to yield the initialized object.

-   Try to be explicit when using functions with args vs. keyword args. For example, in a function `f` with argument `a` and keyword argument `b`, use `f(a_val, b=b_val)`.

-   End unittest files with

    ```python
    if __name__ == '__main__':
      unittest.main()
    ```

-   Use f-strings instead of .format strings. The former is newer and more readable.

-   Follow the LSP! In your tests, you can often ensure LSP compliance by inheriting from a TestCase subclass instead of directly from TestCase.

-   Use relative imports to import modules within this package. We think this is more readable.

-   We generally use the private designation `_` to denote attributes and functions that are not meant for end users to use. You can still access private attributes from one member of acnportal in another part of gym-acnportal; just include a comment overriding style checkers and explaining why the access is needed.

-   More generally, if you're breaking a style convention, add a comment saying why.

-   Include inline comments, but focus your comments on **why** your code needs to do what it does rather than what it does.

Please consider contributing to other ongoing projects in the ACN Research Portal:
<https://github.com/sunash/gym-acnportal>
