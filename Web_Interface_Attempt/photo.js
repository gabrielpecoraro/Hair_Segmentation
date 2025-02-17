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
    alert (fname); 
	if (fname == "") 
	{ 
        document.getElementById("fname").style.display = "Veuillez remplir le champ"
	return false; 
	} 
	return true; 
} 