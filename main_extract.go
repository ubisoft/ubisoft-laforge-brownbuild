package main

// command : go run main_extract.go -proc 5 -path ./dataset/graphviz/ -out ./dataset/graphviz_extracted/
import (
	"flag"
	"fmt"
	"time"
	"io/ioutil"
	"path"
	"regexp"
	"strings"
  "strconv"
  "sync"
	"os"
	"github.com/caneroj1/stemmer"
)

var file_regex = ".*[\\/]((.*_.*_.*_.*_.*_.*)_(.*)_(.*)_([01])(_(.*))?)(-processed)?\\.log"
var stop_words = []string{"a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"}
// Source stop words: http://xpo6.com/list-of-english-stop-words/

type applicableRegex struct {
	regex       *regexp.Regexp
	replacement string
}

func contains(l[]string, s string) bool {
  for _,a := range l {
    for a == s {
      return true
    }
  }
  return false
}

func main() {
	now := time.Now().UTC()
	
  proc := flag.Int("proc", 24, "procs to use")
	pathIn := flag.String("path", "", "the path where you files are")
	pathOut := flag.String("out", "", "path to write the results to")
	
	flag.Parse()
	
	regexes := [16]string{"\n", " ", "(?P<url>https?://[^\\s]+)", "hypothesisurlforge","[^\\s]+[/\\\\][^\\s]+","hypothesispathforge", "[^\\s]+\\.[^\\s]+","hypothesispathforge", "[\\d\\w]*\\w\\d[\\d\\w]*", "hypothesisnumletforge", "[\\d\\w]+\\d\\w[\\d\\w]*", "hypothesisnumletforge", "[_\\W]+", " ", "([A-Z]+)", " $1"}
	
	var applicableRegexes []*applicableRegex
	
	for index := 0; index < len(regexes); index = index + 2 {
		applicableRegexes = append(
			applicableRegexes,
			&applicableRegex{
				regexp.MustCompile(regexes[index]),
				regexes[index+1],
			},
		)
	}
	
	var wg sync.WaitGroup
	
	files := make(chan string)
	
  for index := 0; index < *proc; index++ {
		wg.Add(1)
		go processFiles(&wg, files, applicableRegexes, *pathOut)
	}
	
	filesOnDisk, err := ioutil.ReadDir(*pathIn)
	if err != nil {
		panic(fmt.Sprintf("Can't read path in dir - %s - %v", *pathIn, err))
	}
	
	for _, f := range filesOnDisk {
		files <- *pathIn + f.Name()
	}
	
	
	close(files)
	
  wg.Wait()
	fmt.Println("Done " + *pathOut)

	fmt.Println("--- ", time.Since(now)," ---")
}


func processFiles(wg *sync.WaitGroup, files chan string, applicableRegexes []*applicableRegex, out string) {
	defer wg.Done()
  for file := range files {
		contentBytes, _ := ioutil.ReadFile(file)
		content := string(contentBytes)
    
    pat := regexp.MustCompile(file_regex)
    groups := pat.FindStringSubmatch(file)
    
		for _, regex := range applicableRegexes {
      
			content = regex.regex.ReplaceAllString(content, regex.replacement)
		}
    
		words := stemmer.StemMultiple(strings.Fields(strings.ToLower(content)))
		var words_red = []string{}
	  for i:=0; i < len(words); i++ {
	    if ! contains(stop_words, strings.ToLower(words[i])) && len(words[i]) > 2  {
	      words_red = append(words_red, strings.ToLower(words[i]))
	    }
	  }
	  
		var ngram=2
		var new_content = WordCount(words_red, ngram)
		
		if _, err := os.Stat(out); os.IsNotExist(err) {
			os.Mkdir(out, os.ModePerm)
		}
		ioutil.WriteFile(path.Join(out, groups[1]+ "-processed.csv"), []byte(new_content), 0777)
  }
}

func InterWordCount(words []string, ngram int) string {
	counts := make(map[string]int, len(words)-ngram)
	for _, word := range words {
			counts[strings.ToLower(word)]++
	}
	
	return StringCount(counts)
}

func WordCount(words []string, ngram int) string {
		mem_words := make([]string, len(words))
		copy(mem_words, words)
		
		ret_string := InterWordCount(mem_words,1)
		
		words_ngram := mem_words[0:len(words)]
		
		for i := 1; i < ngram; i++ {
			if len(words) < i {
				return ret_string
			}
			for j := i; j < len(words); j++ {
				words_ngram[j-i] += "_" + words[j]
			}
			ret_string += "#####" + InterWordCount(mem_words[0:len(mem_words)-i], i)
		}
	
    return ret_string
}

func StringCount(m map[string]int) string {
  var ret string = ""
  for k, v := range m {
    ret += k + "," + strconv.Itoa(v) + "\n"
  }
  return ret
}
