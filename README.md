# Deprected
This repository is deprecated and will not be maintained any more. We will soon archive it.

Please use https://github.com/ls1intum/thesis-template-typst based on [Typst](https://typst.app) instead which is easier, faster and more fun!

## thesis-template
A Latex template for your Bachelor's or Master's thesis.

_Please note:_ This is only a template. You have to adapt the template to your thesis and discuss the structure of your thesis with your supervisor!

--- 
## Guidelines 

__Please thorougly read our guidelines and hints on [confluence](https://confluence.ase.in.tum.de/display/EduResStud/How+to+thesis)!__ (TUM Login Required) 

---

## Usage 
### Set thesis metadata 
Fill in your thesis details in the `metadata` file: 
* Degree (Bachelor or Master)
* Your study program
* English and German title
* Advisor and supervisor
* Your name (without e-mail address or matriculation number)
* The start and submission date

### Build PDFs locally 

```bash
latexmk -pdf
```

Clean temp files: 
```
latexmk -CA
```

## Build and release a PDF automatically

This template contains a GitHub workflow that automatically builds the LaTeX document and creates a release on GitHub with the built PDF.
For new repositories generated by this template, the workflow is enabled by default.

Using GitHub Actions is free of charge for public repositories.
To learn more about pricing, please refer to [this documentation on billing](https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions).

### Disable automatic builds

In case you do not want this feature, you can simply remove the `build-release-thesis.yml` file from the `.github/workflows/` folder.

### Add the workflow to existing repositories

You can also use the workflow with an existing repository.
Either copy the file `build-release-thesis.yml` from the folder `.github/workflows/` within this template to the exact same location in your repository or apply a patch with these commands:

>
> ```bash
> # Move into your repository
> cd <REPOSITORY>
>
> # Get the changes and apply them
> curl -L https://github.com/ls1intum/thesis-template/pull/12/commits/0679ed5d48e361edf2866b02f39832e6552d0033.patch | git apply
> git add .github/workflows/build-release-thesis.yml
> git commit -m "Add workflow for automatic LaTeX builds."
> 
> # (Optional) Push the changes
> git push
> ```
