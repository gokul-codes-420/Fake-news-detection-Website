      // ── Truth or Myth ───────────────────────────────────────────────────────
      const tmData = [
        { statement: "Incognito mode completely hides your activity from everyone.", isTruth: false, exp: "Incognito mode only prevents your browser from saving local history. Your ISP, school, or workplace can still track you." },
        { statement: "Mac computers cannot get viruses or malware.", isTruth: false, exp: "While historically less targeted than Windows PCs, Macs can and absolutely do get malware." },
        { statement: "HTTPS sites are always safe and legitimate.", isTruth: false, exp: "HTTPS just means the connection is encrypted. Scammers easily get HTTPS certificates for fake phishing websites." },
        { statement: "You should restart your phone or computer regularly.", isTruth: true, exp: "Restarting helps install pending security patches and OS updates properly." },
        { statement: "Banks will never ask for your OTP or password over a call.", isTruth: true, exp: "True. Official bank reps will never ask you to reveal your OTP, PIN, or password verbally." }
      ];
      
      let tmIndex = 0;
      
      function renderTruthOrMyth() {
        document.getElementById('tm-question-container').classList.remove('hidden');
        document.getElementById('tm-result-container').classList.add('hidden');
        document.getElementById('tm-statement').innerText = tmData[tmIndex].statement;
      }
      
      function checkTruthOrMyth(userSelectedTruth) {
        const item = tmData[tmIndex];
        const isCorrect = (userSelectedTruth === item.isTruth);
        
        document.getElementById('tm-question-container').classList.add('hidden');
        const resContainer = document.getElementById('tm-result-container');
        const resTitle = document.getElementById('tm-result-title');
        
        resContainer.classList.remove('hidden');
        resContainer.className = `mt-2 p-5 rounded-lg border ${isCorrect ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800' : 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'}`;
        
        resTitle.innerText = isCorrect ? "✅ Correct!" : "❌ Incorrect";
        resTitle.className = `font-bold mb-2 ${isCorrect ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'}`;
        
        document.getElementById('tm-result-explanation').innerText = `The answer is ${item.isTruth ? 'Truth' : 'Myth'}. ${item.exp}`;
      }
      
      function nextTruthOrMyth() {
        tmIndex = (tmIndex + 1) % tmData.length;
        renderTruthOrMyth();
      }
      
      renderTruthOrMyth();

    </script>
