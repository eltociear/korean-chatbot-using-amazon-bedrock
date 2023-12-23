const myForm = document.querySelector('#my-form');
const userInput = document.querySelector('#userId');
const convtypeInput = document.querySelector('#convtype');

myForm.addEventListener('submit', onSubmit);

// load userId 
let userId = localStorage.getItem('userId'); // set userID if exists 
if(userId != '') {
    userInput.value = userId;
}

let conversationType = localStorage.getItem('conv_type'); // set conversationType if exists 
if(conversationType != '') {
    convtypeInput.value = conversationType;
}
else {
    convtypeInput.value = "normal"  // general conversation
}

console.log(userInput.value);
console.log(convtypeInput.value);

// provisioning
getProvisioningInfo(userId);

function onSubmit(e) {
    e.preventDefault();
    console.log(userInput.value);
    console.log(convtypeInput.value);

    localStorage.setItem('userId',userInput.value);
    console.log('Save Profile> userId:', userInput.value)    

    localStorage.setItem('conv_type',convtypeInput.value);
    console.log('Save Profile> conv_type:', convtypeInput.value)

    window.location.href = "chat.html";
}

function getProvisioningInfo(userId) {
    const uri = "provisioning";
    const xhr = new XMLHttpRequest();

    xhr.open("POST", uri, true);
    xhr.onreadystatechange = () => {
        if (xhr.readyState === 4 && xhr.status === 200) {
            let response = JSON.parse(xhr.responseText);
            let provisioning_info = JSON.parse(response['info']);
            console.log("provisioning info: " + JSON.stringify(provisioning_info));
                        
            let connection_url = provisioning_info.connection_url;
            console.log("connection_url: ", connection_url);

            localStorage.setItem('connection_url',connection_url);
        }
    };

    var requestObj = {
        "userId": userId
    }
    console.log("request: " + JSON.stringify(requestObj));

    var blob = new Blob([JSON.stringify(requestObj)], {type: 'application/json'});

    xhr.send(blob);   
}
