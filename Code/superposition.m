% Charger l'image originale
image_originale = imread('python.jpg');

% Redimensionner l'image en 256x256 pixels
image_redimensionnee = imresize(image_originale, [256, 256]);

% Afficher l'image redimensionnée
imshow(image_redimensionnee);

% Charger le masque des cheveux en niveaux de gris
masque_cheveux = imread('Predicted_red.png');

% Convertir les images en double pour effectuer des calculs
image_originale_double = im2double(image_redimensionnee);
masque_cheveux_double = im2double(masque_cheveux);

% Superposer le masque des cheveux sur l'image originale
image_resultat = bsxfun(@times, image_redimensionnee, masque_cheveux_double);

% Afficher l'image résultante
imshow(image_resultat);
