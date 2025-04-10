# Nivel de energía en escala del 1 al 10

# Grupo con desayuno (15 personas)
desayuno <- c(8, 9, 8, 7, 9, 8, 9, 7, 8, 8, 9, 7, 8, 9, 8)

# Grupo sin desayuno (15 personas)
sin_desayuno <- c(5, 4, 6, 5, 4, 6, 5, 5, 4, 6, 5, 4, 5, 5, 4)

# Medias
mean(desayuno)         # Promedio del grupo con desayuno
mean(sin_desayuno)     # Promedio del grupo sin desayuno

# Desviaciones estándar
sd(desayuno)
sd(sin_desayuno)

#varianza
cv <-sd(desayuno)/mean(sin_desayuno)
cv*100

#cv
t.test(desayuno, sin_desayuno)
