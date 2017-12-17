//const uri = "http://localhost:5000/";
const uri = "http://footballanalysis.herokuapp.com/";
const today = new Date();
const ms_in_day = 1000*60*60*24;
var user_history;
var cache_life = new Date();


//Reset cache for testing
//localStorage.clear();

$("#favorites_form").submit(function(e) {
    e.preventDefault();
});

function update_cache(){
    var uri_cache = uri+"max_difference/";
    $.get(uri_cache+"points", function(points){
        $.get(uri_cache+"yards", function(yards){
            $.get(uri_cache+"pass_yards",function(pass_yards){
                $.get(uri_cache+"rush_yards", function(rush_yards){
                    var cache_data = {
                        "points":points,
                        "yards_overall":yards,
                        "yards_passing":pass_yards,
                        "rushing_yards":rush_yards,
                        "expiry":new Date()
                    }
                    first_run = false;
                    user_history = cache_data;
                    localStorage.setItem("history", JSON.stringify(user_history));
                });
            });
        });
    });
}

function check_cache(){
    if(!user_history){ //First run, by default, no cache
        var cached_string = localStorage.getItem("history");
        if(!cached_string){ //Nothing in the cache yet
            update_cache();
        } else {
            user_history = JSON.parse(cached_string);
            cache_life = new Date(user_history['expiry']);
        }
    }

    if((today.getTime()-cache_life.getTime())/ms_in_day > 7){ //Cache expires after a week
        update_cache();
    }
}

function compare_teams(form){
    check_cache();
    var team_a = form.team_a.value;
    var team_b = form.team_b.value;

    $.get(uri+"team_a="+team_a+"&team_b="+team_b, function(data){
        console.log(data);
        var pass = data.pass_coords;
        var points = data.points_coords;
        var rush = data.rush_coords;
        var yards = data.yards_coords;
        $("#team_name_A").text(team_a);
        $("#team_name_B").text(team_b);
        analyze_numbers(team_a, team_b, "points",points);
        analyze_numbers(team_a, team_b, "yards_passing",pass);
        analyze_numbers(team_a, team_b, "rushing_yards",rush);
        analyze_numbers(team_a, team_b, "yards_overall",yards);
        
    });
}
const less_good_def = "Less good defensively";
const less_good_of = "Less good offensively";
const good_def = "Better defensively";
const good_of = "Better offensively";
const equal = "The two teams are equal";
function analyze_numbers(team_a, team_b, stat_name, data){
    var output_string_A = "";
    var output_string_B = "";

    var best_offensively;
    var best_defesively;

    if(data[0] < 0){
        output_string_B += less_good_def+". ";
        output_string_A += good_def+". ";
        best_defensively = team_a;
    } else if(data[0] > 0) {
        output_string_A += less_good_def+". ";
        output_string_B += good_def+". ";
        best_defensively = team_b;  
    } else{
        output_string_A += equal+". ";
        output_string_B += equal+". ";
        best_defensively = "Equal";
    }

    if(data[1] < 0){
        output_string_B += good_of+". ";
        output_string_A += less_good_of+". ";
        best_offensively = team_b;
    } else if(data[1] > 0) {
        output_string_A += good_of+". ";
        output_string_B += less_good_of+". ";
        best_offensively = team_a;
    } else{
        output_string_B += equal+". ";
        output_string_A += equal+". ";
        best_offensively = "Equal";
    }

    $("#"+stat_name+"_A").text(output_string_A);
    $("#"+stat_name+"_B").text(output_string_B);

    diff_o = parseInt(Math.abs(data[0])/(user_history[stat_name]["offense"])*100);
    diff_d = parseInt(Math.abs(data[1])/(user_history[stat_name]["defense"])*100);
    $("#bar_"+stat_name+"_o").css("width", diff_o+"%").attr('aria-valuenow', diff_o).text(best_offensively);
    $("#bar_"+stat_name+"_d").css("width", diff_d+"%").attr('aria-valuenow', diff_d).text(best_defensively);

    var x = parseInt(data[0]);
    var y = parseInt(data[1]);
    $("#"+stat_name).text("Raw "+stat_name+": "+x+" | "+y);
}