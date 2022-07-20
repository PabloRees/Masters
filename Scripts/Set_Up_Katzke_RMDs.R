devtools::install_github('Nicktz/fmxdat')
install.packages(c('tinytex', 'rmarkdown'))
tinytex::install_tinytex()
devtools::install_github("Nicktz/Texevier")
devtools::install(xtable)
library(glue)


fmxdat::make_project(Mac=T, Open = T)


CHOSEN_LOCATION <- "/Users/pablo/Desktop/DS_Test/"
fmxdat::make_project(Mac=T,Open=T)

Texevier::create_template(directory = glue::glue("{CHOSEN_LOCATION}Solution/19119461/"), template_name = "Question4")
Texevier::create_template(directory = glue::glue("{CHOSEN_LOCATION}Solution/19119461/"), template_name = "Question2")
Texevier::create_template(directory = glue::glue("{CHOSEN_LOCATION}Solution/19119461/"), template_name = "Question3")

