document.getElementById("fname").style.display = "block";
function toggleNameInput(type) {
    if(type.value == '1'){
        document.getElementById("fname").style.display = "block";
    }
    else{
        document.getElementById("fname").style.display = "none";
    }
}

function submitForm() 
{ 
var fname = document.getElementById("fname").value;
var mcharge = document.getElementById("mcharge").value;
var empty = document.getElementById("empty");
alert(fname);
if (fname == "") 
	{ 
    alert("Veuillez Remplir le Champ")  
	return false;  
	}
alert("ok");
if(mcharge == "1") 
    {
        document.location.href = "/Users/gabrielpecoraro/Desktop/Cours_ENSEIRB/2A/S8/Projet_Thematique/pr214_HC/hc/raspberry/photoself.html";
    }





}   