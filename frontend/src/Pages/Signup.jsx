// In SignupForm or separate OTPForm
import { sendSignInLinkToEmail, isSignInWithEmailLink, signInWithEmailLink } from "firebase/auth";

function OTPForm() {
  const [email, setEmail] = useState("");
  const handleSendLink = () => {
    const actionCodeSettings = { url: "https://yourapp.com/otp-verify", handleCodeInApp: true };
    sendSignInLinkToEmail(auth, email, actionCodeSettings)
      .then(()=> localStorage.setItem("emailForSignIn", email));
  };
  useEffect(() => {
    if (isSignInWithEmailLink(auth, window.location.href)) {
      let email = localStorage.getItem("emailForSignIn");
      signInWithEmailLink(auth, email, window.location.href)
        .then(()=>{/* redirect to Dashboard */});
    }
  }, []);
  return (
    <>
      <input value={email} onChange={e=>setEmail(e.target.value)} placeholder="Email for OTP"/>
      <button onClick={handleSendLink}>Send OTP</button>
    </>
  );
}
